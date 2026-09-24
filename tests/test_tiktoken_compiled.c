#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_tokenizer.h"

int main(int argc,char **argv)
{
    SparkTokenizer raw,compiled;
    SparkTokenizerTiktokenRanksConfiguration r = {SPARK_TOKENIZER_ABI_VERSION,sizeof(r),0,0,0};
    SparkTokenizerCompiledFileConfiguration c = {SPARK_TOKENIZER_ABI_VERSION,sizeof(c),0,0,0};
    uint32_t seed = 17u,iteration;
    assert(argc == 3);
    r.ranks_path = argv[1]; c.compiled_tokenizer_path = argv[2];
    assert(SparkTokenizerLoadTiktokenRanks(&raw,&r) == SPARK_STATUS_OK);
    assert(SparkTokenizerLoadCompiledFile(&compiled,&c) == SPARK_STATUS_OK);
    for (iteration=0; iteration<1000u; iteration++)
    {
        char prompt[1024],decoded[2048];
        uint32_t a[2048],b[2048],i,bytes;
        SparkTokenizerEncoding ea,eb;
        strcpy(prompt,"Hello world! 日本語 中文 한국어 123456789\n");
        for (i=(uint32_t)strlen(prompt); i<300u; i++)
        { seed=seed*1664525u+1013904223u; prompt[i]=(char)(32u+(seed>>24u)%95u); }
        prompt[300]='\0';
        SparkTokenizerEncodingReset(&ea); SparkTokenizerEncodingReset(&eb);
        ea.token_ids=a; ea.token_capacity=2048; eb.token_ids=b; eb.token_capacity=2048;
        assert(SparkTokenizerEncodeUtf8(&raw,prompt,300u,0,&ea) == SPARK_STATUS_OK);
        assert(SparkTokenizerEncodeUtf8(&compiled,prompt,300u,0,&eb) == SPARK_STATUS_OK);
        assert(ea.token_count == eb.token_count && memcmp(a,b,ea.token_count*sizeof(*a)) == 0);
        assert(SparkTokenizerDecodeTokenIds(&compiled,b,eb.token_count,0,decoded,sizeof(decoded),&bytes) == SPARK_STATUS_OK);
        assert(bytes == 300u && memcmp(prompt,decoded,bytes) == 0);
    }
    for (iteration=0; iteration<compiled.special_token_count; iteration++)
    {
        SparkTokenizerSpecialToken *special=&compiled.special_tokens[iteration];
        SparkTokenizerEncoding encoding;
        uint32_t ids[64],bytes;
        char decoded[1024];
        SparkTokenizerEncodingReset(&encoding); encoding.token_ids=ids; encoding.token_capacity=64;
        assert(SparkTokenizerEncodeUtf8(&compiled,special->text,special->text_bytes,0,&encoding) == SPARK_STATUS_OK);
        assert(encoding.token_count == 1u && ids[0] == special->token_id);
        assert(SparkTokenizerDecodeTokenIds(&compiled,ids,1,0,decoded,sizeof(decoded),&bytes) == SPARK_STATUS_OK);
        assert(bytes == special->text_bytes && memcmp(decoded,special->text,bytes) == 0);
    }
    printf("PASS 1000 raw/compiled token comparisons and %u added-token round trips\n",compiled.special_token_count);
    SparkTokenizerDestroy(&raw); SparkTokenizerDestroy(&compiled);
    return 0;
}
