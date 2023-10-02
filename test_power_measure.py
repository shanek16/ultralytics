from jtop import jtop
import threading
import time

def power_monitoring(samples):
    with jtop() as jetson:
        while not exit_flag:
            power = jetson.power['tot']['power']
            samples.append(power)
            total_memory = jetson.memory['RAM']['used'] + jetson.memory['SWAP']['used']
            memory_samples.append(total_memory)
            time.sleep(1)  # sample every second
exit_flag = False
samples = []
memory_samples = []

# Start the power monitoring thread
monitor_thread = threading.Thread(target=power_monitoring, args=(samples,))
monitor_thread.start()
time.sleep(3)
exit_flag = True
monitor_thread.join()

# Calculate the total power consumed
total_power_consumed = sum(samples) - samples[0]  # subtracting the initial power reading
print(f"Total power consumed: {total_power_consumed:.2f} mW")
# Calculate the max and min memory used
memory_usage_difference = max(memory_samples) - min(memory_samples)
print(f"Memory usage difference (max - min): {memory_usage_difference:.2f} MB")