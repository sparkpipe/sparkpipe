from pathlib import Path

ep = Path("/Users/mac/lane-glm53/ring/transport/tp_device_collective.c")
es = ep.open().read()

# Replace SendPersistent with SendFixed in the BuildSend path
old_send = """        if (status == SPARK_STATUS_OK)
            status = SparkHiddenTransportSend(
                implementation->send_sessions[route_index],&send_packet);"""
assert es.count(old_send) == 1, "send anchor: %d" % es.count(old_send)
es = es.replace(old_send, """        if (status == SPARK_STATUS_OK)
        {
            if (operation->algorithm_kind ==
                    SPARK_TP_DEVICE_COLLECTIVE_LITERAL_RING_KIND &&
                implementation->transport_library.transport_interface
                    .send_fixed != 0)
            {
                status =
                    implementation->transport_library.transport_interface
                        .send_fixed(
                    implementation->send_sessions[route_index],
                    send_packet.hidden_bf16,
                    (uint64_t)send_packet.active_sequence_count *
                        send_packet.bytes_per_sequence,
                    (uint32_t)operation->ordinal);
            }
            else
            {
                status = SparkHiddenTransportSend(
                    implementation->send_sessions[route_index],
                    &send_packet);
            }
        }""")

ep.open("w").write(es)
print("SendFixed integrated into literal ring send path")
