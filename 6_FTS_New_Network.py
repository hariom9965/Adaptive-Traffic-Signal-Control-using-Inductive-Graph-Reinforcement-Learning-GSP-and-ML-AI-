# Fixed-time SUMO run (phases & timings set in SUMO config) with metrics & plots
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'New_Network.sumocfg',
    '--step-length', '1',
    '--delay', '10',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Variables
# -------------------------
TOTAL_STEPS = 10000    # simulation steps

# -------------------------
# Intersection definitions
# -------------------------
INTERSECTIONS = {
    "Node0": {
        "detectors": [
            "Node0_10_SB_0", "Node0_10_SB_1",
            "Node4_0_NB_0",  "Node4_0_NB_1",
            "Node0_1_WB_0",  "Node0_1_WB_1",
            "Node0_16_EB_1", "Node0_16_EB_0",
        ],
        "neighbors": ["Node10", "Node1", "Node16"],
    },
    "Node1": {
        "detectors": [
            "Node0_1_EB_0",  "Node0_1_EB_1",
            "Node1_6_SB_0",  "Node1_6_SB_1",
            "Node3_1_NB_0",  "Node3_1_NB_1",
            "Node1_2_WB_0",  "Node1_2_WB_1",
        ],
        "neighbors": ["Node6", "Node0", "Node2", "Node3"],
    },
    "Node2": {
        "detectors": [
            "Node1_2_EB_0",  "Node1_2_EB_1",
            "Node2_7_SB_0",  "Node2_7_SB_1",
            "Node5_2_NB_0",  "Node5_2_NB_1",
            "Node2_12_WB_0", "Node2_12_WB_1",
        ],
        "neighbors": ["Node1", "Node7", "Node5", "Node12"],
    },
    "Node3": {
        "detectors": [
            "Node4_3_EB_0",  "Node4_3_EB_1",
            "Node3_1_NB_0",  "Node3_1_NB_1",
            "Node3_5_WB_0",  "Node3_5_WB_1",
            "Node3_18_NB_0", "Node3_18_NB_1",
        ],
        "neighbors": ["Node4", "Node1", "Node5", "Node18"],
    },
    "Node4": {
        "detectors": [
            "Node4_0_SB_0",  "Node4_0_SB_1",
            "Node13_4_NB_0", "Node13_4_NB_1",
            "Node4_3_WB_0",  "Node4_3_WB_1",
            "Node4_17_EB_1", "Node4_17_EB_0",
        ],
        "neighbors": ["Node0", "Node13", "Node3", "Node17"],
    },
    "Node5": {
        "detectors": [
            "Node3_5_EB_0",  "Node3_5_EB_1",
            "Node5_2_SB_0",  "Node5_2_SB_1",
            "Node5_11_WB_0", "Node5_11_WB_1",
            "Node5_19_NB_0", "Node5_19_NB_1",
        ],
        "neighbors": ["Node3", "Node2", "Node11", "Node19"],
    },
    "Node6": {
        "detectors": [
            "Node6_15_EB_0",  "Node6_15_EB_1",
            "Node1_6_NB_0",   "Node1_6_NB_1",
            "Node6_7_WB_0",   "Node6_7_WB_1",
            "Node6_9_SB_1",   "Node6_9_SB_0",
        ],
        "neighbors": ["Node9", "Node1", "Node7", "Node15"],
    },
    "Node7": {
        "detectors": [
            "Node6_7_EB_0",  "Node6_7_EB_1",
            "Node2_7_NB_0",  "Node2_7_NB_1",
            "Node7_8_WB_0",  "Node7_8_WB_1",
            "Node7_14_SB_1", "Node7_14_SB_0",
        ],
        "neighbors": ["Node6", "Node8", "Node2", "Node14"],
    },
}

# Traffic light id — update to match your primary controlled intersection
traffic_light_id = "Node2"
Total_Intersections=8

# -------------------------
# Helper functions
# -------------------------
def get_queue_length(detector_id):
    """Return queue length measured by a lane-area detector (vehicles in last step)."""
    try:
        return traci.lanearea.getLastStepVehicleNumber(detector_id)
    except Exception:
        return 0

def get_current_phase(tls_id):
    """Return current phase index of traffic light."""
    try:
        return traci.trafficlight.getPhase(tls_id)
    except Exception:
        return -1

def get_total_queue_all_intersections():
    """Sum queue lengths across all detectors in all intersections."""
    total = 0
    for node, data in INTERSECTIONS.items():
        for det in data["detectors"]:
            total += get_queue_length(det)
    return total

def get_queue_per_intersection():
    """Return dict of {node_name: total_queue} for all intersections."""
    result = {}
    for node, data in INTERSECTIONS.items():
        result[node] = sum(get_queue_length(det) for det in data["detectors"])
    return result

def read_state():
    """
    Read all detector queues across all intersections + current phase of primary TLS.
    Returns a flat list: [all queue values..., current_phase]
    """
    queues = []
    for node, data in INTERSECTIONS.items():
        for det in data["detectors"]:
            queues.append(get_queue_length(det))
    current_phase = get_current_phase(traffic_light_id)
    return queues + [current_phase]

# -------------------------
# Data recording lists
# -------------------------
step_history = []
queue_history = []
reward_history = []
throughput_history = []
waiting_time_history = []

# Per-intersection queue history for detailed plots
per_intersection_queue_history = {node: [] for node in INTERSECTIONS}
per_intersection_step_history = []

print("\n=== Starting Fixed-Time Simulation (phases set in SUMO) ===")
print(f"Monitoring {len(INTERSECTIONS)} intersections, "
      f"{sum(len(d['detectors']) for d in INTERSECTIONS.values())} detectors total.\n")

# Subscribe to waiting time for all vehicles automatically as they depart.
# We use traci.vehicle.subscribe on new vehicles each step — but the fastest
# approach is to use context subscriptions on the simulation domain.
WAITING_TIME_VAR = traci.constants.VAR_WAITING_TIME

def get_avg_waiting_time():
    """
    Compute average waiting time across all active vehicles efficiently.
    Uses a single getIDList() call + subscribed variable cache instead of
    individual getWaitingTime() calls per vehicle.
    """
    try:
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        # Subscribe any vehicle we haven't seen yet (subscription results are
        # returned in the same step's response — near-zero overhead vs raw calls)
        for vid in veh_ids:
            traci.vehicle.subscribe(vid, [WAITING_TIME_VAR])
        total_wait = 0.0
        count = 0
        for vid in veh_ids:
            result = traci.vehicle.getSubscriptionResults(vid)
            if result and WAITING_TIME_VAR in result:
                total_wait += result[WAITING_TIME_VAR]
                count += 1
        return (total_wait / count) if count > 0 else 0.0
    except Exception:
        return 0.0



# -------------------------
# Main simulation loop (no phase changes)
# -------------------------
for step in range(TOTAL_STEPS):

    # Advance simulation one step
    traci.simulationStep()

    # Throughput: arrived vehicles this step
    try:
        throughput = traci.simulation.getArrivedNumber()
    except Exception:
        throughput = 0
    throughput_history.append(throughput)

    # Waiting time via subscriptions (fast — cached by SUMO after simulationStep)
    avg_wait = get_avg_waiting_time()
    waiting_time_history.append(avg_wait)

    # Only do the expensive detector reads at sample points
    if step % 100 == 0:
        per_node_q = get_queue_per_intersection()
        total_queue = sum(per_node_q.values())
        avg_queue_per_intersection = total_queue / len(INTERSECTIONS)  # ADD THIS
        reward = -float(total_queue)

        print(f"Step {step}: total_queue={total_queue}, avg_queue_per_intersection={avg_queue_per_intersection:.2f}, avg_wait={avg_wait:.2f}s")  # UPDATE
        print("  Per-intersection queues: " +
              ", ".join(f"{n}={q}" for n, q in per_node_q.items()))

        step_history.append(step)
        queue_history.append(avg_queue_per_intersection)  # CHANGE: was total_queue
        reward_history.append(reward)
        per_intersection_step_history.append(step)
        for node in INTERSECTIONS:
            per_intersection_queue_history[node].append(per_node_q[node])

# -------------------------
# Close SUMO connection
# -------------------------
traci.close()



# -------------------------
# Final summaries
# -------------------------
avg_queue_over_samples = float(np.mean(queue_history)) if queue_history else 0.0
total_throughput = int(np.sum(throughput_history))
mean_waiting_time = float(np.mean(waiting_time_history)) if waiting_time_history else 0.0

print("\n=== SUMMARY METRICS ===")
print(f"Total throughput counted (sum of per-step arrivals): {total_throughput}")
print(f"Average waiting time (mean across all steps): {mean_waiting_time:.2f} sec")
print(f"Average queue length (mean across sampled steps): {avg_queue_over_samples:.2f} vehicles")

# -------------------------
# Plots
# -------------------------
"""
# 1. Total queue length over time (sampled)
plt.figure(figsize=(10, 5))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue (all intersections)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length (vehicles)")
plt.title("Fixed-Time: Total Queue Length over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Per-intersection queue length over time
plt.figure(figsize=(12, 6))
for node, q_vals in per_intersection_queue_history.items():
    plt.plot(per_intersection_step_history, q_vals, marker='.', linestyle='-', label=node)
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.title("Fixed-Time: Queue Length per Intersection over Time")
plt.grid(True)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

# 3. Average waiting time (windowed)
window = 100
avg_waiting_time_windows = [
    np.mean(waiting_time_history[i: i + window])
    for i in range(0, len(waiting_time_history), window)
]

plt.figure(figsize=(10, 5))
plt.plot(avg_waiting_time_windows, marker='o', linestyle='-', color='darkorange')
plt.xlabel("Window Index (each = 100 steps)")
plt.ylabel("Avg Waiting Time (sec)")
plt.title("Fixed-Time: Average Waiting Time (every 100 steps)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Cumulative throughput
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(throughput_history), linestyle='-', color='green')
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Arrived Vehicles")
plt.title("Fixed-Time: Cumulative Throughput")
plt.grid(True)
plt.tight_layout()
plt.show()
"""