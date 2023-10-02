from jtop import jtop


if __name__ == "__main__":

    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        while jetson.ok():
            # # CPU
            # print('*** CPUs ***')
            # print(jetson.cpu)
            # CPU
            print('*** Memory ***')
            # print(jetson.memory)
            print(jetson.memory['RAM']['used'])
            print(jetson.memory['SWAP']['used'])
            # # GPU
            # print('*** GPU ***')
            # print(jetson.gpu)
            # # Engines
            # print('*** engine ***')
            # print(jetson.engine)
            # # nvpmodel
            # print('*** NV Power Model ***')
            # print(jetson.nvpmodel)
            # # jetson_clocks
            # print('*** jetson_clocks ***')
            # print(jetson.jetson_clocks)
            # # Status disk
            # print('*** disk ***')
            # print(jetson.disk)
            # # Status fans
            # print('*** fan ***')
            # print(jetson.fan)
            # # uptime
            # print('*** uptime ***')
            # print(jetson.uptime)
            # # local interfaces
            # print('*** local interfaces ***')
            # print(jetson.local_interfaces)
            # # Temperature
            # print('*** temperature ***')
            # print(jetson.temperature)
            # # Power
            # print('*** power ***')
            # print(jetson.power)
            # # print(jetson.power['tot']['power'])
# EOF