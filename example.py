from avp_stream import VisionProStreamer
avp_ip = "10.12.102.110"   # example IP 
s = VisionProStreamer(ip = avp_ip)

while True:
    r = s.latest
    # print(r['head'], r['right_wrist'], r['right_fingers'])
    print(r['right_wrist'])
