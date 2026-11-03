#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#define MODEL_RANKS 16u
#define MODEL_OPS 12u
#define MODEL_EVENTS 512u

enum { OP_WHOLE, OP_PIPE, OP_PAIR_GATHER, OP_PAIR_SCATTER };
enum { EVENT_PUBLISH, EVENT_ARRIVE, EVENT_READ };
enum { SCHEDULE_PRODUCTION, SCHEDULE_EAGER };

typedef struct ModelEvent
{
    uint8_t kind;
    uint8_t op;
    uint16_t round;
} ModelEvent;

typedef struct ModelOp
{
    uint32_t kind;
    uint32_t chunks;
    uint32_t rounds;
    uint64_t first_tag;
} ModelOp;

typedef struct ModelCell
{
    uint64_t tag;
    uint32_t live;
} ModelCell;

typedef struct Model
{
    uint32_t degree,victim,op_count,length,violations,deadlock;
    ModelOp ops[MODEL_OPS];
    ModelEvent program[MODEL_EVENTS];
    uint32_t cursor[MODEL_RANKS];
    ModelCell data[MODEL_RANKS][MODEL_RANKS][2][2];
    ModelCell tail[MODEL_RANKS][MODEL_RANKS][2][2];
} Model;

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond,name) do { test_checks++; if ( !(cond) ) { test_failures++; fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); } } while (0)

static uint32_t ModelSlot(uint64_t tag)
{
    return (uint32_t)((tag - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
}

static uint32_t ModelHalf(uint64_t tag)
{
    return SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag)) != 0u ? 1u : 0u;
}

static uint32_t ModelTail(const ModelOp *op,uint64_t tag)
{
    const uint32_t half = op->kind == OP_WHOLE ? SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE : SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag);
    return SPARK_WEIGHTD_MESH_TAIL_OFFSET(half) == (uint64_t)SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u ? 1u : 0u;
}

static void ModelEmit(Model *model,uint32_t kind,uint32_t op,uint64_t round)
{
    if ( model->length < MODEL_EVENTS )
    {
        model->program[model->length].kind = (uint8_t)kind;
        model->program[model->length].op = (uint8_t)op;
        model->program[model->length].round = (uint16_t)round;
    }
    model->length++;
}

static void ModelProgram(Model *model,uint32_t schedule)
{
    uint64_t tag = 1u;
    for ( uint32_t index = 0u; index < model->op_count; index++ )
    {
        ModelOp *op = &model->ops[index];
        const uint32_t pair = op->kind == OP_PAIR_GATHER || op->kind == OP_PAIR_SCATTER ? 1u : 0u;
        const uint64_t lead = pair != 0u && schedule == SCHEDULE_EAGER ? 2u : pair != 0u ? SPARK_TP_MESH_PIPELINE_LEAD : 1u;
        op->rounds = op->kind == OP_WHOLE ? 1u : op->kind == OP_PIPE ? op->chunks :
            op->kind == OP_PAIR_GATHER ? op->chunks + SPARK_TP_MESH_PAIR_FIRST_GATHER_EXTRA_ROUNDS : op->chunks + 2u;
        op->first_tag = tag;
        tag += op->rounds;
        if ( op->kind == OP_WHOLE )
        {
            ModelEmit(model,EVENT_PUBLISH,index,0u);
            ModelEmit(model,EVENT_ARRIVE,index,0u);
            ModelEmit(model,EVENT_READ,index,0u);
            continue;
        }
        for ( uint64_t round = 0u; round < lead; round++ )
            ModelEmit(model,EVENT_PUBLISH,index,round);
        for ( uint64_t round = 0u; round < op->rounds; round++ )
        {
            const uint64_t early = pair != 0u && schedule == SCHEDULE_EAGER ? SPARK_TP_MESH_PIPELINE_NONE : SparkTpMeshPipelineEarly(round,op->rounds);
            const uint64_t late = pair != 0u && schedule == SCHEDULE_EAGER ? (round + 2u < op->rounds ? round + 2u : SPARK_TP_MESH_PIPELINE_NONE) :
                SparkTpMeshPipelineLate(round,op->rounds);
            ModelEmit(model,EVENT_ARRIVE,index,round);
            if ( early != SPARK_TP_MESH_PIPELINE_NONE )
                ModelEmit(model,EVENT_PUBLISH,index,early);
            ModelEmit(model,EVENT_READ,index,round);
            if ( late != SPARK_TP_MESH_PIPELINE_NONE )
                ModelEmit(model,EVENT_PUBLISH,index,late);
        }
    }
}

static void ModelStore(Model *model,ModelCell *cell,uint64_t tag,uint32_t live)
{
    if ( cell->live != 0u )
        model->violations++;
    cell->tag = tag;
    cell->live = live;
}

static uint32_t ModelReads(const ModelOp *op,uint32_t receiver,uint32_t sender,uint64_t round)
{
    const uint32_t same = ((receiver ^ sender) & 1u) == 0u ? 1u : 0u;
    if ( receiver == sender )
        return 0u;
    if ( op->kind == OP_PAIR_GATHER )
        return same != 0u ? (round < op->chunks ? 1u : 0u) : (round >= 2u && round < op->chunks + 2u ? 1u : 0u);
    if ( op->kind == OP_PAIR_SCATTER )
        return same != 0u ? (round >= 2u ? 1u : 0u) : (round < op->chunks ? 1u : 0u);
    return 1u;
}

static void ModelPublish(Model *model,uint32_t sender,const ModelOp *op,uint64_t round)
{
    const uint64_t tag = op->first_tag + round;
    const uint32_t slot = ModelSlot(tag),half = ModelHalf(tag),tail = ModelTail(op,tag);
    for ( uint32_t receiver = 0u; receiver < model->degree; receiver++ )
    {
        const uint32_t same = ((receiver ^ sender) & 1u) == 0u ? 1u : 0u;
        if ( receiver == sender )
            continue;
        if ( op->kind == OP_WHOLE )
        {
            ModelStore(model,&model->data[receiver][sender][slot][0],tag,1u);
            ModelStore(model,&model->data[receiver][sender][slot][1],tag,1u);
        }
        else if ( op->kind == OP_PIPE )
            ModelStore(model,&model->data[receiver][sender][slot][half],tag,1u);
        else
        {
            const uint32_t group = op->kind == OP_PAIR_GATHER ? (round < op->chunks || round >= op->chunks + 2u ? 1u : 0u) : (round >= 2u ? 1u : 0u);
            const uint32_t pair = op->kind == OP_PAIR_GATHER ? (round >= 2u && round < op->chunks + 2u ? 1u : 0u) : (round < op->chunks ? 1u : 0u);
            if ( same != 0u && group != 0u )
                ModelStore(model,&model->data[receiver][sender][slot][half],tag,ModelReads(op,receiver,sender,round));
            if ( receiver == (sender ^ 1u) && pair != 0u )
                for ( uint32_t segment = receiver & 1u; segment < model->degree; segment += 2u )
                    ModelStore(model,&model->data[receiver][segment ^ 1u][slot][half],tag,ModelReads(op,receiver,segment ^ 1u,round));
        }
        ModelStore(model,&model->tail[receiver][sender][slot][tail],tag,1u);
    }
}

static uint32_t ModelArrived(const Model *model,uint32_t rank,const ModelOp *op,uint64_t round)
{
    const uint64_t tag = op->first_tag + round;
    for ( uint32_t sender = 0u; sender < model->degree; sender++ )
        if ( sender != rank && model->tail[rank][sender][ModelSlot(tag)][ModelTail(op,tag)].tag != tag )
            return 0u;
    return 1u;
}

static void ModelRun(Model *model,uint32_t rank)
{
    const ModelEvent *event = &model->program[model->cursor[rank]++];
    const ModelOp *op = &model->ops[event->op];
    const uint64_t tag = op->first_tag + event->round;
    const uint32_t slot = ModelSlot(tag),half = ModelHalf(tag);
    if ( event->kind == EVENT_PUBLISH )
    {
        ModelPublish(model,rank,op,event->round);
        return;
    }
    for ( uint32_t sender = 0u; sender < model->degree; sender++ )
    {
        if ( sender == rank )
            continue;
        if ( event->kind == EVENT_ARRIVE )
        {
            model->tail[rank][sender][slot][ModelTail(op,tag)].live = 0u;
            continue;
        }
        if ( ModelReads(op,rank,sender,event->round) == 0u )
            continue;
        for ( uint32_t part = 0u; part < 2u; part++ )
        {
            ModelCell *cell = &model->data[rank][sender][slot][part];
            if ( op->kind != OP_WHOLE && part != half )
                continue;
            if ( cell->tag != tag )
                model->violations++;
            cell->live = 0u;
        }
    }
}

static uint32_t ModelReady(const Model *model,uint32_t rank)
{
    const ModelEvent *event;
    if ( model->cursor[rank] >= model->length )
        return 0u;
    event = &model->program[model->cursor[rank]];
    return event->kind != EVENT_ARRIVE || ModelArrived(model,rank,&model->ops[event->op],event->round) != 0u ? 1u : 0u;
}

static uint32_t ModelEager(const Model *model,uint32_t rank)
{
    return rank != model->victim || model->program[model->cursor[rank]].kind == EVENT_PUBLISH ? 1u : 0u;
}

static void ModelSimulate(Model *model)
{
    for ( ;; )
    {
        uint32_t rank,ran = 0u,done = 1u;
        for ( rank = 0u; rank < model->degree; rank++ )
        {
            done &= model->cursor[rank] >= model->length ? 1u : 0u;
            while ( ModelReady(model,rank) != 0u && ModelEager(model,rank) != 0u )
            {
                ModelRun(model,rank);
                ran = 1u;
            }
        }
        if ( ran != 0u )
            continue;
        if ( ModelReady(model,model->victim) != 0u )
        {
            ModelRun(model,model->victim);
            continue;
        }
        model->deadlock = done != 0u ? 0u : 1u;
        return;
    }
}

static uint32_t ModelCase(uint32_t degree,uint32_t victim,const uint32_t *kinds,uint32_t count,uint32_t chunks,uint32_t lead_whole,uint32_t schedule)
{
    static Model model;
    memset(&model,0,sizeof(model));
    if ( lead_whole + count > MODEL_OPS )
        return UINT32_MAX;
    model.degree = degree;
    model.victim = victim;
    for ( uint32_t index = 0u; index < lead_whole; index++ )
        model.ops[model.op_count++].kind = OP_WHOLE;
    for ( uint32_t index = 0u; index < count; index++ )
    {
        model.ops[model.op_count].kind = kinds[index];
        model.ops[model.op_count++].chunks = chunks;
    }
    ModelProgram(&model,schedule);
    if ( model.length > MODEL_EVENTS )
        return UINT32_MAX;
    ModelSimulate(&model);
    return model.violations + model.deadlock;
}

static uint32_t ModelSweep(const uint32_t *kinds,uint32_t count,uint32_t schedule)
{
    static const uint32_t degrees[3] = {4u,8u,16u};
    uint32_t faults = 0u;
    for ( uint32_t which = 0u; which < 3u; which++ )
    {
        const uint32_t degree = degrees[which],victims[3] = {0u,1u,degree - 1u};
        for ( uint32_t victim = 0u; victim < 3u; victim++ )
            for ( uint32_t chunks = 3u; chunks <= 5u; chunks++ )
                for ( uint32_t lead_whole = 0u; lead_whole < 4u; lead_whole++ )
                    faults += ModelCase(degree,victims[victim],kinds,count,chunks,lead_whole,schedule) != 0u ? 1u : 0u;
    }
    return faults;
}

int main(void)
{
    static const uint32_t whole_gather[3] = {OP_WHOLE,OP_PAIR_GATHER,OP_WHOLE};
    static const uint32_t whole_scatter[3] = {OP_WHOLE,OP_PAIR_SCATTER,OP_WHOLE};
    static const uint32_t whole_pipe[3] = {OP_WHOLE,OP_PIPE,OP_WHOLE};
    static const uint32_t mixed[7] = {OP_PAIR_GATHER,OP_PAIR_GATHER,OP_PAIR_SCATTER,OP_PIPE,OP_PAIR_GATHER,OP_WHOLE,OP_PAIR_SCATTER};
    CHECK(SparkTpMeshPipelineEarly(0u,3u) == 1u && SparkTpMeshPipelineEarly(1u,3u) == SPARK_TP_MESH_PIPELINE_NONE &&
        SparkTpMeshPipelineLate(0u,7u) == 2u && SparkTpMeshPipelineLate(3u,7u) == 5u && SparkTpMeshPipelineLate(4u,7u) == SPARK_TP_MESH_PIPELINE_NONE &&
        SparkTpMeshPipelineLate(5u,7u) == 6u && SparkTpMeshPipelineLate(6u,7u) == SPARK_TP_MESH_PIPELINE_NONE,
        "a pipelined round publishes round 1 after round 0 arrives and its last round after the one before it arrives");
    CHECK(ModelSweep(whole_gather,3u,SCHEDULE_EAGER) != 0u && ModelSweep(whole_scatter,3u,SCHEDULE_EAGER) != 0u,
        "publishing two pair-first rounds before the first arrival lets a fast rank overwrite a slow rank's unread whole-slot round");
    CHECK(ModelSweep(whole_gather,3u,SCHEDULE_PRODUCTION) == 0u,
        "a pair-first all-gather between whole-slot rounds never overwrites a slot or tail a slow rank has not read, at 4, 8 and 16 ranks");
    CHECK(ModelSweep(whole_scatter,3u,SCHEDULE_PRODUCTION) == 0u,
        "a pair-first reduce-scatter between whole-slot rounds never overwrites a slot or tail a slow rank has not read, at 4, 8 and 16 ranks");
    CHECK(ModelSweep(whole_pipe,3u,SCHEDULE_PRODUCTION) == 0u,
        "a plain pipelined round between whole-slot rounds never overwrites unread data");
    CHECK(ModelSweep(mixed,7u,SCHEDULE_PRODUCTION) == 0u,
        "back-to-back pair-first, pipelined and whole-slot rounds never overwrite unread data");
    fprintf(stderr,"test_tp_mesh_pipeline_boundaries: %u checks, %u failures\n",test_checks,test_failures);
    return test_failures != 0u ? 1 : 0;
}
