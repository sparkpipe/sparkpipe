/* Unit test for the tree-capable acceptance resolution in the neutral
 * speculation-policy core: root reject, accept-to-depth, branch selection,
 * duplicate-sibling rejection, and chain-as-degenerate-tree parity. */
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_speculation_policy.h"

#define SPARK_TEST_TREE_VOCAB_SIZE 1000u

static void SparkTestTreeResolveRootReject(void)
{
    const uint32_t draft_tokens[2] = { 10u, 11u };
    const uint32_t parents[2] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u
    };
    const uint32_t verifier_tokens[3] = { 77u, 11u, 12u };
    SparkSpeculationPolicyVerifyResult result;

    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        parents,
        2u,
        verifier_tokens,
        3u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_OK);
    assert((result.flags &
        SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    assert(result.proposed_token_count == 2u);
    assert(result.accepted_draft_token_count == 0u);
    assert(result.committed_token_count == 1u);
    assert(result.fallback_token_id == 77u);
}

static void SparkTestTreeResolveAcceptToDepth(void)
{
    const uint32_t draft_tokens[3] = { 10u, 11u, 12u };
    const uint32_t parents[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 1u
    };
    const uint32_t verifier_tokens[4] = { 10u, 11u, 88u, 13u };
    SparkSpeculationPolicyVerifyResult result;

    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_OK);
    assert((result.flags &
        SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    assert(result.accepted_draft_token_count == 2u);
    assert(result.committed_token_count == 3u);
    assert(result.fallback_token_id == 88u);
}

static void SparkTestTreeResolveBranchSelection(void)
{
    /* nodes 0,1 are root children; node 2 hangs off node 1. The verifier
     * picks node 1's token at the root row, so the accepted path runs
     * through the second branch. */
    const uint32_t draft_tokens[3] = { 10u, 20u, 21u };
    const uint32_t parents[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT,
        SPARK_SPECULATION_PARENT_INDEX_ROOT,
        1u
    };
    const uint32_t verifier_tokens[4] = { 20u, 11u, 21u, 22u };
    SparkSpeculationPolicyVerifyResult result;

    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_OK);
    assert((result.flags &
        SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    assert(result.accepted_draft_token_count == 2u);
    assert(result.committed_token_count == 3u);
    assert(result.fallback_token_id == 22u);
}

static void SparkTestTreeResolveAcceptsAllWithBonus(void)
{
    const uint32_t draft_tokens[3] = { 10u, 11u, 12u };
    const uint32_t parents[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 1u
    };
    const uint32_t verifier_tokens[4] = { 10u, 11u, 12u, 13u };
    SparkSpeculationPolicyVerifyResult result;

    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_OK);
    assert((result.flags &
        SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u);
    assert(result.accepted_draft_token_count == 3u);
    assert(result.committed_token_count == 4u);
    assert(result.fallback_token_id == 13u);
}

static void SparkTestTreeResolveChainParity(void)
{
    const uint32_t draft_tokens[4] = { 11u, 12u, 13u, 14u };
    const uint32_t parents[4] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 1u, 2u
    };
    const uint32_t verifier_tokens[5] = { 11u, 12u, 99u, 99u, 42u };
    SparkSpeculationPolicyVerifyResult chain_result;
    SparkSpeculationPolicyVerifyResult tree_result;

    assert(SparkSpeculationPolicyResolveVerifierTokens(
        draft_tokens,
        4u,
        verifier_tokens,
        5u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &chain_result) == SPARK_STATUS_OK);
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        parents,
        4u,
        verifier_tokens,
        5u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &tree_result) == SPARK_STATUS_OK);
    assert(memcmp(&chain_result, &tree_result, sizeof(chain_result)) == 0);
    assert(chain_result.accepted_draft_token_count == 2u);
    assert(chain_result.committed_token_count == 3u);
    assert(chain_result.fallback_token_id == 99u);
}

static void SparkTestTreeResolveRejectsMalformedTrees(void)
{
    const uint32_t draft_tokens[3] = { 10u, 11u, 12u };
    const uint32_t duplicate_root_siblings[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 0u
    };
    const uint32_t duplicate_root_tokens[3] = { 10u, 11u, 11u };
    const uint32_t duplicate_sibling_parents[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 0u
    };
    const uint32_t self_parent[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 0u, 2u
    };
    const uint32_t forward_parent[3] = {
        SPARK_SPECULATION_PARENT_INDEX_ROOT, 2u, 1u
    };
    const uint32_t verifier_tokens[4] = { 10u, 11u, 12u, 13u };
    const uint32_t out_of_vocab_draft[3] = { 10u, 11u, 1000u };
    SparkSpeculationPolicyVerifyResult result;
    const uint32_t *null_parents = 0;

    /* duplicate sibling tokens under one parent */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        duplicate_root_tokens,
        duplicate_sibling_parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    /* a node that is its own parent */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        self_parent,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    /* a parent that does not precede its child */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        forward_parent,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    /* an explicit tree must supply the bonus row (count + 1 rows) */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        duplicate_root_siblings,
        3u,
        verifier_tokens,
        3u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    /* draft token at the vocab bound fails loudly */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        out_of_vocab_draft,
        null_parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    /* null arrays and zero counts fail loudly */
    assert(SparkSpeculationPolicyResolveVerifierTree(
        0,
        null_parents,
        3u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        null_parents,
        0u,
        verifier_tokens,
        4u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkSpeculationPolicyResolveVerifierTree(
        draft_tokens,
        null_parents,
        3u,
        verifier_tokens,
        4u,
        0u,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkSpeculationPolicyResolveVerifierTokens(
        draft_tokens,
        3u,
        verifier_tokens,
        5u,
        SPARK_TEST_TREE_VOCAB_SIZE,
        &result) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestTreeResolveRootReject();
    SparkTestTreeResolveAcceptToDepth();
    SparkTestTreeResolveBranchSelection();
    SparkTestTreeResolveAcceptsAllWithBonus();
    SparkTestTreeResolveChainParity();
    SparkTestTreeResolveRejectsMalformedTrees();
    return 0;
}
