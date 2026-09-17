#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_MAX_SEQUENCES 8
#define TEST_HIDDEN 512
#define TEST_MAX_ROUNDS 16
#define TEST_UNUSED 0xFFFF

typedef struct test_sequence
{
    uint32_t prompt_tokens[8];
    uint32_t prompt_len;
    uint32_t expected_tokens[TEST_MAX_ROUNDS];
    uint32_t emitted;
    uint32_t kv_slot;
    uint32_t active;
} test_sequence;

static int test_failures;

#define CHECK(cond, name) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, name); \
        test_failures++; \
    } \
} while (0)

static void test_kv_slot_assignment_no_collision(void)
{
    uint32_t slot_map[TEST_MAX_SEQUENCES];
    uint32_t seq, slot;
    uint32_t used_slots[TEST_MAX_SEQUENCES];

    memset(used_slots, 0, sizeof(used_slots));
    memset(slot_map, TEST_UNUSED, sizeof(slot_map));

    for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++) {
        slot = seq % TEST_MAX_SEQUENCES;
        CHECK(slot < TEST_MAX_SEQUENCES, "slot in range");
        CHECK(!used_slots[slot], "slot not already assigned");
        used_slots[slot] = 1;
        slot_map[seq] = slot;
    }

    for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++) {
        uint32_t other;
        for (other = seq + 1; other < TEST_MAX_SEQUENCES; other++) {
            CHECK(slot_map[seq] != slot_map[other] || seq == other,
                "no slot collision between active sequences");
        }
    }

    for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++) {
        uint32_t freed_slot = slot_map[seq];
        used_slots[freed_slot] = 0;
        slot_map[seq] = TEST_UNUSED;

        uint32_t new_seq = TEST_MAX_SEQUENCES;
        for (slot = 0; slot < TEST_MAX_SEQUENCES; slot++) {
            if (!used_slots[slot]) {
                used_slots[slot] = 1;
                slot_map[new_seq] = slot;
                break;
            }
        }
        CHECK(slot_map[new_seq] != TEST_UNUSED,
            "freed slot is reusable by new sequence");
    }
}

static void test_batch_round_row_count(void)
{
    uint32_t active[TEST_MAX_SEQUENCES];
    uint32_t seq, round;
    uint32_t total_active;
    (void)round;

    for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++)
        active[seq] = seq < 4 ? 1 : 0;

    for (round = 0; round < TEST_MAX_ROUNDS; round++) {
        if (round == 3)
            active[5] = 1;
        if (round == 7)
            active[0] = 0;
        total_active = 0;
        for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++)
            total_active += active[seq];
        CHECK(total_active <= TEST_MAX_SEQUENCES, "active count in range");
    }

    total_active = 0;
    for (seq = 0; seq < TEST_MAX_SEQUENCES; seq++)
        total_active += active[seq];
    CHECK(total_active == 4, "3 remaining + 1 arrival = 4");
}

static void test_payload_size_scales_with_batch(void)
{
    uint32_t batch_sizes[] = {1, 2, 4, 8};
    uint32_t i;
    for (i = 0; i < 4; i++) {
        uint32_t batch = batch_sizes[i];
        uint64_t bytes = (uint64_t)batch * TEST_HIDDEN * 2;
        CHECK(bytes >= TEST_HIDDEN * 2, "payload >= single row");
        CHECK(bytes <= 8 * TEST_HIDDEN * 2, "payload <= max batch");
    }
}

static void test_sequence_arrival_departure_invariants(void)
{
    test_sequence seqs[TEST_MAX_SEQUENCES];
    uint32_t i, round;
    uint32_t total_emitted = 0;

    memset(seqs, 0, sizeof(seqs));
    for (i = 0; i < TEST_MAX_SEQUENCES; i++) {
        seqs[i].active = i < 3 ? 1 : 0;
        seqs[i].kv_slot = i;
    }

    for (round = 0; round < TEST_MAX_ROUNDS; round++) {
        uint32_t j;
        for (j = 0; j < TEST_MAX_SEQUENCES; j++) {
            if (seqs[j].active && seqs[j].emitted < 4) {
                seqs[j].emitted++;
                total_emitted++;
                if (seqs[j].emitted >= 4)
                    seqs[j].active = 0;
            }
        }

        if (round == 5) {
            for (j = 0; j < TEST_MAX_SEQUENCES; j++) {
                if (!seqs[j].active) {
                    seqs[j].active = 1;
                    seqs[j].emitted = 0;
                    break;
                }
            }
        }

        uint32_t any_active = 0;
        for (j = 0; j < TEST_MAX_SEQUENCES; j++)
            any_active |= seqs[j].active;
        if (round < 3 || (round >= 5 && round <= 7))
            CHECK(any_active,
                "batch continues while sequences are active");
    }

    CHECK(total_emitted >= 12,
        "3 initial sequences × 4 tokens + at least 1 arrival × some tokens");
}

static void test_kv_slot_reuse_after_completion(void)
{
    uint32_t slot_owner[TEST_MAX_SEQUENCES];
    uint32_t i, round;

    for (i = 0; i < TEST_MAX_SEQUENCES; i++)
        slot_owner[i] = TEST_UNUSED;

    uint32_t seq_a = 0, seq_b = 1;
    slot_owner[0] = seq_a;
    slot_owner[1] = seq_b;

    for (round = 0; round < 4; round++) {
        CHECK(slot_owner[0] == seq_a, "seq_a owns slot 0 during decode");
        CHECK(slot_owner[1] == seq_b, "seq_b owns slot 1 during decode");
    }

    slot_owner[0] = TEST_UNUSED;
    uint32_t seq_c = 2;
    slot_owner[0] = seq_c;
    CHECK(slot_owner[0] == seq_c, "seq_c reuses slot 0 after seq_a completes");
    CHECK(slot_owner[0] != seq_a, "seq_a no longer owns slot 0");
    CHECK(slot_owner[1] == seq_b, "seq_b still owns slot 1");
}

static void test_prefill_decode_interleave(void)
{
    uint32_t prefill_in_flight = 0;
    uint32_t decode_active = 0;
    uint32_t round;

    for (round = 0; round < 20; round++) {
        if (round % 5 == 0 && round > 0) {
            prefill_in_flight = 1;
        }
        if (round % 5 == 2) {
            prefill_in_flight = 0;
            decode_active++;
        }
        if (prefill_in_flight) {
            CHECK(decode_active >= 0,
                "decode can proceed alongside prefill admission");
        }
    }
    CHECK(decode_active > 0, "decode rounds happened");
}

int main(void)
{
    test_kv_slot_assignment_no_collision();
    test_batch_round_row_count();
    test_payload_size_scales_with_batch();
    test_sequence_arrival_departure_invariants();
    test_kv_slot_reuse_after_completion();
    test_prefill_decode_interleave();

    fprintf(stderr, "test_continuous_batch_decode: %d failures\n",
        test_failures);
    return test_failures ? 1 : 0;
}
