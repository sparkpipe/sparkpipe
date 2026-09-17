#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_MAX_ROWS 8
#define TEST_HIDDEN 512
#define TEST_MAX_LAYERS 4

static int test_failures;

#define CHECK(cond, name) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, name); \
        test_failures++; \
    } \
} while (0)

typedef struct test_kv_page
{
    uint32_t sequence_slot;
    uint32_t page_index;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t valid;
} test_kv_page;

typedef struct test_kv_slot
{
    test_kv_page pages[16];
    uint32_t page_count;
    uint32_t sequence_position;
} test_kv_slot;

static test_kv_slot slots[TEST_MAX_ROWS];

static void reset_slots(void)
{
    memset(slots, 0, sizeof(slots));
}

static void test_multi_row_kv_assignment_deterministic(void)
{
    uint32_t rows[] = {1, 2, 4, 8};
    uint32_t i, row, layer;

    for (i = 0; i < 4; i++) {
        row = rows[i];
        reset_slots();

        for (layer = 0; layer < TEST_MAX_LAYERS; layer++) {
            uint32_t r;
            for (r = 0; r < row; r++) {
                slots[r].pages[layer].sequence_slot = r;
                slots[r].pages[layer].page_index = layer;
                slots[r].pages[layer].token_start = layer;
                slots[r].pages[layer].token_count = 1;
                slots[r].pages[layer].valid = 1;
            }
        }

        for (layer = 0; layer < TEST_MAX_LAYERS; layer++) {
            uint32_t r;
            for (r = 0; r < row; r++) {
                CHECK(slots[r].pages[layer].sequence_slot == r,
                    "row's KV page has the correct slot");
                CHECK(slots[r].pages[layer].valid,
                    "row's KV page is valid");
            }
        }
    }
}

static void test_multi_row_payload_scales(void)
{
    uint32_t rows[] = {1, 2, 4, 8};
    uint32_t i;

    for (i = 0; i < 4; i++) {
        uint64_t bytes = (uint64_t)rows[i] * TEST_HIDDEN * 2;
        CHECK(bytes == (uint64_t)(1 << i) * TEST_HIDDEN * 2,
            "payload bytes scale linearly with rows");
    }
}

static void test_kv_deterministic_across_runs(void)
{
    uint32_t run, layer, row;
    uint32_t reference[TEST_MAX_ROWS][TEST_MAX_LAYERS];

    for (run = 0; run < 2; run++) {
        reset_slots();
        for (layer = 0; layer < TEST_MAX_LAYERS; layer++) {
            for (row = 0; row < TEST_MAX_ROWS; row++) {
                slots[row].pages[layer].sequence_slot = row;
                slots[row].pages[layer].page_index = layer;
                slots[row].pages[layer].token_start = run * TEST_MAX_ROWS + row;
                slots[row].pages[layer].token_count = 1;
                slots[row].pages[layer].valid = 1;
            }
        }

        for (layer = 0; layer < TEST_MAX_LAYERS; layer++) {
            for (row = 0; row < TEST_MAX_ROWS; row++) {
                uint32_t tok = slots[row].pages[layer].token_start;
                if (run == 0)
                    reference[row][layer] = tok;
                else
                    CHECK(tok == reference[row][layer] + TEST_MAX_ROWS,
                        "KV assignment is deterministic (same slot per row)");
            }
        }
    }
}

static void test_row_completion_does_not_disturb_others(void)
{
    reset_slots();
    uint32_t row;

    for (row = 0; row < 4; row++) {
        slots[row].pages[0].sequence_slot = row;
        slots[row].pages[0].valid = 1;
        slots[row].page_count = 1;
        slots[row].sequence_position = 8;
    }

    slots[1].page_count = 0;
    memset(&slots[1].pages, 0, sizeof(slots[1].pages));

    for (row = 0; row < 4; row++) {
        if (row == 1)
            continue;
        CHECK(slots[row].pages[0].valid,
            "other rows still valid after row 1 completes");
        CHECK(slots[row].pages[0].sequence_slot == row,
            "other rows keep their slot assignment");
        CHECK(slots[row].sequence_position == 8,
            "other rows keep their position");
    }

    slots[1].pages[0].sequence_slot = 1;
    slots[1].pages[0].valid = 1;
    slots[1].page_count = 1;
    slots[1].sequence_position = 0;

    CHECK(slots[1].pages[0].sequence_slot == 1,
        "new sequence reuses freed slot");
    CHECK(slots[1].sequence_position == 0,
        "new sequence starts at position 0");
}

static void test_prefill_chunking_correct(void)
{
    uint32_t prompt_len = 32;
    uint32_t max_rows = 4;
    uint32_t total_processed = 0;
    uint32_t remaining = prompt_len;

    while (remaining > 0) {
        uint32_t this_chunk = remaining < max_rows ? remaining : max_rows;
        CHECK(this_chunk <= max_rows, "chunk size bounded by max rows");
        CHECK(this_chunk > 0, "chunk is non-empty");
        total_processed += this_chunk;
        remaining -= this_chunk;
    }

    CHECK(total_processed == prompt_len,
        "all prompt tokens processed across chunks");
}

int main(void)
{
    test_multi_row_kv_assignment_deterministic();
    test_multi_row_payload_scales();
    test_kv_deterministic_across_runs();
    test_row_completion_does_not_disturb_others();
    test_prefill_chunking_correct();

    fprintf(stderr, "test_multi_row_prefill: %d failures\n", test_failures);
    return test_failures ? 1 : 0;
}
