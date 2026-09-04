p = '/Users/mac/lane-glm53/ring/transport/tp_device_collective.c'
s = open(p).read()

old = '''    for (lane_index = 0u; lane_index < 2u; lane_index++)
    {
        chunk = SparkTpDeviceCollectiveLiteralRingSendChunk(
            implementation->collective,phase_index,lane_index);'''
new = '''    for (lane_index = operation->lane_id;
         lane_index < 1u + operation->lane_id;
         lane_index++)
    {
        chunk = SparkTpDeviceCollectiveLiteralRingSendChunk(
            implementation->collective,phase_index,lane_index);'''
assert s.count(old) == 1, 'send pack anchor'
s = s.replace(old, new)

old2 = '''    phase = operation->current_step;
    for (lane_index = 0u; lane_index < 2u; lane_index++)
    {
        chunk = SparkTpDeviceCollectiveLiteralRingReceiveChunk(
            implementation->collective,phase,lane_index);'''
assert s.count(old2) == 1, 'consume anchor'
s = s.replace(old2, '''    phase = operation->current_step;
    for (lane_index = operation->lane_id;
         lane_index < 1u + operation->lane_id;
         lane_index++)
    {
        chunk = SparkTpDeviceCollectiveLiteralRingReceiveChunk(
            implementation->collective,phase,lane_index);''')

open(p, 'w').write(s)
print('single-lane loops installed')
