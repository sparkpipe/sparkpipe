p = '/Users/mac/lane-glm53/ring/transport/tp_device_collective.c'
s = open(p).read()

old = '''    operation->current_step = 0u;
    operation->lane_step[0] = 0u;
    operation->lane_step[1] = 0u;
    operation->lane_receive_pending = 3u;'''
if s.count(old) == 1:
    s = s.replace(old, '''    operation->current_step = 0u;
    operation->lane_id = credit_index % 2u;''')
    print('lane init replaced')

old2 = '''    uint32_t current_step;
    uint32_t lane_step[2];
    uint32_t lane_receive_pending;
    uint32_t algorithm_kind;'''
if s.count(old2) == 1:
    s = s.replace(old2, '''    uint32_t current_step;
    uint32_t lane_id;
    uint32_t algorithm_kind;''')
    print('struct fields replaced')

old3 = '''    operation->current_step = 0u;
    operation->algorithm_kind = SparkTpDeviceCollectiveSelectAlgorithm('''
if s.count(old3) == 1:
    s = s.replace(old3, '''    operation->current_step = 0u;
    operation->lane_id = credit_index % 2u;
    operation->algorithm_kind = SparkTpDeviceCollectiveSelectAlgorithm(''')
    print('submit init patched')

open(p, 'w').write(s)
print('done')
