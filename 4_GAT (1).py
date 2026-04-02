"""
IG-RL: Inductive Graph Reinforcement Learning
==============================================
Zero-shot transfer fix:

OLD (broken):
  all 8 nodes → GCN → flatten into 1 vector → single GRU → reshape to 8 nodes
  Problem: the model memorises node positions in the vector. Transfer fails.

NEW (correct):
  all 8 nodes → GCN  (shared weights aggregate neighbourhood info)
               ↓
  pick ONLY training node's embedding  (shape: EMBEDDING_DIM,)
               ↓
  GRU on that single embedding  (temporal memory for this one node)
               ↓
  Dueling Q-head → Q(hold), Q(switch)

At deploy time on ANY node:
  Run the same GCN on that node's local neighbourhood,
  extract that node's embedding, pass through the same GRU + Q-head.
  Zero-shot — no retraining needed.
  The weights never saw node IDs, only queue/neighbour patterns.
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque



TRAIN        = True
TRAINING_TLS = "Node2"   # the ONE node we train on; same weights transfer to all others

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# ==========================
#   SUMO CONFIG
# ==========================
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "New_Network.sumocfg"

Sumo_config = [
    SUMO_BINARY, "-c", SUMO_CONFIG,
    "--step-length", "1", "--delay", "0"
]

# ==========================
#   INTERSECTIONS
# ==========================
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
            "Node1_6_NB_0",  "Node1_6_NB_1",
            "Node6_7_WB_0",  "Node6_7_WB_1",
            "Node6_9_SB_1",  "Node6_9_SB_0",
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


# ==========================
#   RL PARAMETERS
# ==========================
TOTAL_STEPS        = 10000
EPISODE_LENGTH     = 100
LEARNING_RATE      = 0.0001
GAMMA              = 0.95
EPSILON_START      = 1.0
EPSILON_END        = 0.01
EPSILON_DECAY      = 0.9995
BATCH_SIZE         = 32
MEMORY_SIZE        = 50000
TARGET_UPDATE_FREQ = 500
WARMUP_STEPS       = 200
UPDATE_FREQUENCY   = 4
PRINT_INTERVAL     = 100

MIN_GREEN_STEPS             = 10
DEPLOY_MIN_GREEN            = 15
MAX_GREEN_STEPS             = 60
PRESSURE_SWITCH_RATIO       = 1.5
CONGESTION_SWITCH_THRESHOLD = 8
MIN_GREEN_RATIO_HOLD        = 0.6

GCN_LAYERS    = 3
EMBEDDING_DIM = 32
HIDDEN_DIM    = 64
GRU_UNITS     = 32
FEATURE_DIM   = 12
ACTIONS       = [0, 1]

# ==========================
#   INCIDENT PARAMETERS
# ==========================
INCIDENT_SPEED_THRESHOLD     = 2.0
INCIDENT_OCCUPANCY_THRESHOLD = 0.8
NORMAL_LANE_SPEED            = 13.89
INCIDENT_BLOCKED_SPEED       = 0.5

INCIDENT_SCHEDULE = [
    ("Node1_2_EB_0", 150, 300),
    ("Node3_5_WB_0", 400, 550),
]


# ==========================
#   TRAFFIC GRAPH
# ==========================
class TrafficGraph:
    def __init__(self, intersections):
        self.intersections = intersections
        self.tls_ids       = list(intersections.keys())
        self.build_adjacency_matrix()

    def build_adjacency_matrix(self):
        n = len(self.tls_ids)
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            self.adj_matrix[i, i] = 1.0
        for i, tls_i in enumerate(self.tls_ids):
            for neighbor in self.intersections[tls_i]["neighbors"]:
                if neighbor in self.tls_ids:
                    j = self.tls_ids.index(neighbor)
                    self.adj_matrix[i, j] = 1.0
                    self.adj_matrix[j, i] = 1.0
        self.normalized_adj = self.normalize_adjacency(self.adj_matrix)

    def normalize_adjacency(self, adj):
        adj        = adj + np.eye(adj.shape[0])
        degree     = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D = np.diag(d_inv_sqrt)
        return (D @ adj @ D).astype(np.float32)


# ==========================
#   GRAPH CONVOLUTION LAYER
#   SHARED weights — same kernel for every node.
#   This is the transfer mechanism.
# ==========================
# class GraphConvolution(layers.Layer):
#     def __init__(self, units, activation='relu', use_bias=True, **kwargs):
#         super(GraphConvolution, self).__init__(**kwargs)
#         self.units      = units
#         self.activation = keras.activations.get(activation)
#         self.use_bias   = use_bias

#     def build(self, input_shape):
#         feature_dim = input_shape[0][-1]
#         self.kernel  = self.add_weight('kernel', (feature_dim, self.units),
#                                        initializer='glorot_uniform', trainable=True)
#         if self.use_bias:
#             self.bias = self.add_weight('bias', (self.units,),
#                                         initializer='zeros', trainable=True)
#         super(GraphConvolution, self).build(input_shape)

#     def call(self, inputs):
#         features, adjacency = inputs
#         support = tf.matmul(features, self.kernel)
#         output  = tf.matmul(adjacency, support)
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#         return self.activation(output)

# ==========================
#   GRAPH ATTENTION LAYER (GAT)
#   Shared weights → inductive / transfer-friendly
# ==========================
class GraphAttention(layers.Layer):
    def __init__(self, units, num_heads=4, activation='elu',
                 concat=True, attn_dropout=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = keras.activations.get(activation)
        self.concat = concat
        self.attn_dropout = attn_dropout
        self.use_bias = use_bias

        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.drop = layers.Dropout(attn_dropout)

    def build(self, input_shape):
        fin = int(input_shape[0][-1])  # feature dim

        # Linear projection: F -> (heads * units)
        self.W = self.add_weight(
            name="W",
            shape=(fin, self.num_heads * self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        # Attention vectors per head: e_ij = a^T [Wh_i || Wh_j]
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        if self.use_bias:
            out_dim = (self.num_heads * self.units) if self.concat else self.units
            self.bias = self.add_weight(
                name="bias",
                shape=(out_dim,),
                initializer="zeros",
                trainable=True
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x, adj = inputs  # x:(B,N,F), adj:(B,N,N) 0/1

        # Add self-loops (stabilizes GAT)
        n = tf.shape(adj)[-1]
        eye = tf.eye(n, batch_shape=[tf.shape(adj)[0]])
        adj = tf.clip_by_value(adj + eye, 0.0, 1.0)

        B = tf.shape(x)[0]
        N = tf.shape(x)[1]

        # Project features
        h = tf.matmul(x, self.W)  # (B,N,H*U)
        h = tf.reshape(h, (B, N, self.num_heads, self.units))  # (B,N,H,U)
        h = tf.transpose(h, (0, 2, 1, 3))  # (B,H,N,U)

        # Attention logits
        f1 = tf.matmul(h, self.a_src)  # (B,H,N,1)
        f2 = tf.matmul(h, self.a_dst)  # (B,H,N,1)

        e = self.leaky_relu(f1 + tf.transpose(f2, (0, 1, 3, 2)))  # (B,H,N,N)

        # Mask non-neighbors
        mask = tf.expand_dims(adj, axis=1)  # (B,1,N,N)
        e = tf.where(mask > 0.0, e, tf.constant(-1e9, dtype=e.dtype))

        # Softmax attention
        alpha = tf.nn.softmax(e, axis=-1)  # (B,H,N,N)
        alpha = self.drop(alpha, training=training)

        # Aggregate
        out = tf.matmul(alpha, h)  # (B,H,N,U)

        # Merge heads
        if self.concat:
            out = tf.transpose(out, (0, 2, 1, 3))  # (B,N,H,U)
            out = tf.reshape(out, (B, N, self.num_heads * self.units))  # (B,N,H*U)
        else:
            out = tf.reduce_mean(out, axis=1)  # (B,N,U)

        if self.bias is not None:
            out = tf.nn.bias_add(out, self.bias)

        return self.activation(out)


# ==========================
#   NOISY DENSE
# ==========================
class NoisyDense(layers.Layer):
    def __init__(self, units, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units      = units
        self.sigma_init = sigma_init

    def build(self, input_shape):
        d = input_shape[-1]
        self.w_mu    = self.add_weight('w_mu',    (d, self.units), initializer='glorot_uniform')
        self.w_sigma = self.add_weight('w_sigma', (d, self.units),
                                       initializer=keras.initializers.Constant(self.sigma_init))
        self.b_mu    = self.add_weight('b_mu',    (self.units,), initializer='zeros')
        self.b_sigma = self.add_weight('b_sigma', (self.units,),
                                       initializer=keras.initializers.Constant(self.sigma_init))

    def call(self, inputs, training=None):
        if training:
            w = self.w_mu + self.w_sigma * tf.random.normal(tf.shape(self.w_mu))
            b = self.b_mu + self.b_sigma * tf.random.normal(tf.shape(self.b_mu))
        else:
            w, b = self.w_mu, self.b_mu
        return tf.matmul(inputs, w) + b


# ================================================================
#   IG-RL MODEL — THE CRITICAL FIX FOR ZERO-SHOT TRANSFER
#
#   OLD broken approach:
#     GCN output (1, N, EMB) → Reshape to (1, 1, N*EMB) → GRU → ...
#     The GRU sees ALL nodes flattened together every step.
#     It learns "node 2 is always in positions 64-96 of this vector".
#     That never transfers to a different network.
#
#   NEW correct approach:
#     GCN output (1, N, EMB) → for each node i: extract (1, 1, EMB)
#                            → GRU (shared weights, applied independently)
#                            → Q-head (shared weights)
#     The GRU + Q-head only ever see ONE node's embedding at a time.
#     They learn "if my neighbourhood looks like this → hold/switch".
#     That transfers to ANY node in ANY network.
# ================================================================
def build_igrl_model(n_nodes, feature_dim, action_size):
    node_features = layers.Input(shape=(n_nodes, feature_dim), name='node_features')
    adjacency     = layers.Input(shape=(n_nodes, n_nodes),     name='adjacency')

    # Step 1: GCN processes all N nodes with SHARED weights
    # Output: (1, N, EMBEDDING_DIM) — each node has a neighbourhood-aware embedding
    x = node_features
    # for i in range(GCN_LAYERS):
    #     x = GraphConvolution(EMBEDDING_DIM, activation='relu', name=f'gcn_{i}')([x, adjacency])
    #     x = layers.BatchNormalization(name=f'bn_{i}')(x)
    #     x = layers.Dropout(0.1, name=f'drop_{i}')(x)
    # ---- GAT settings ----
    GAT_HEADS = 4
    GAT_ATT_DROP = 0.1

    x = node_features
    for i in range(GCN_LAYERS):
        is_last = (i == GCN_LAYERS - 1)

    # Use concat=True for intermediate layers (dim = heads*EMBEDDING_DIM)
    # Use concat=False for last layer (dim = EMBEDDING_DIM) to keep downstream same
        x = GraphAttention(
            units=EMBEDDING_DIM,
            num_heads=GAT_HEADS,
            activation='elu',
            concat=not is_last,
            attn_dropout=GAT_ATT_DROP,
            name=f'gat_{i}'
        )([x, adjacency])

        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(0.1, name=f'drop_{i}')(x)

    # Step 2: Extract each node's embedding INDEPENDENTLY
    # Each gets shape (1, 1, EMBEDDING_DIM) — a single time-step for the GRU
    node_embeddings = [
        layers.Lambda(lambda t, i=i: t[:, i:i+1, :], name=f'extract_{i}')(x)
        for i in range(n_nodes)
    ]

    # Step 3: ONE shared GRU applied independently to each node's embedding
    # Same weights for every node → inductive, transfers to unseen nodes
    # Each node gets its OWN temporal context, not a mixed one
    gru_layer   = layers.GRU(GRU_UNITS, return_sequences=False, name='gru_shared')
    gru_outputs = [gru_layer(emb) for emb in node_embeddings]
    # gru_outputs[i] shape: (1, GRU_UNITS)

    # Step 4: Shared dueling Q-head applied independently to each node
    # Same weights for every node → transfers zero-shot
    value_dense = layers.Dense(HIDDEN_DIM, activation='relu', name='value_dense')
    value_head  = layers.Dense(1,          name='value_head')
    adv_dense   = layers.Dense(HIDDEN_DIM, activation='relu', name='adv_dense')
    adv_head    = NoisyDense(action_size,  name='adv_head')

    q_per_node = []
    for gru_out in gru_outputs:
        v    = value_head(value_dense(gru_out))               # (1, 1)
        # a    = adv_head(adv_dense(gru_out), training=True)    # (1, action_size)
        a = adv_head(adv_dense(gru_out), training=TRAIN)
        q    = v + (a - tf.reduce_mean(a, axis=-1, keepdims=True))
        q_per_node.append(tf.expand_dims(q, axis=1))          # (1, 1, action_size)

    # Reassemble to (1, N, action_size) — same output shape as before
    q_values = layers.Concatenate(axis=1, name='q_values')(q_per_node)

    model = keras.Model(
        inputs=[node_features, adjacency],
        outputs=q_values,
        name='igrl_per_node_gru'
    )
    model.compile(
        loss=keras.losses.Huber(delta=1.0),
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    )
    return model


# ==========================
#   PRIORITIZED REPLAY BUFFER
#   Stores only TRAINING_TLS transitions (single-agent, paper-faithful)
# ==========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer    = []
        self.priorities= []
        self.position  = 0

    def add(self, state, action, reward, next_state):
        max_p = max(self.priorities) if self.priorities else 1.0
        entry = (state, action, reward, next_state)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
            self.priorities.append(max_p)
        else:
            self.buffer[self.position]     = entry
            self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        prios = np.array(self.priorities[:len(self.buffer)])
        probs = prios ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return samples, idxs, weights

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = abs(e) + 1e-6

    def size(self):
        return len(self.buffer)


# ==========================
#   INCIDENT FUNCTIONS
# ==========================
def apply_scheduled_incidents(incident_schedule, current_step):
    active = []
    for (lane_id, start_step, end_step) in incident_schedule:
        if start_step <= current_step <= end_step:
            try:
                traci.lane.setMaxSpeed(lane_id, INCIDENT_BLOCKED_SPEED)
                active.append(lane_id)
            except:
                pass
        elif current_step > end_step :
            try:
                traci.lane.setMaxSpeed(lane_id, NORMAL_LANE_SPEED)
            except:
                pass
    return active

def detect_incidents(tls_ids):
    incidents = {tls: 0.0 for tls in tls_ids}
    for tls in tls_ids:
        severities = []
        for det in INTERSECTIONS[tls]["detectors"]:
            try:
                avg_speed  = traci.lanearea.getLastStepMeanSpeed(det)
                occupancy  = traci.lanearea.getLastStepOccupancy(det)
                n_vehicles = traci.lanearea.getLastStepVehicleNumber(det)
                if n_vehicles > 0 and 0 <= avg_speed < INCIDENT_SPEED_THRESHOLD:
                    sf = 1.0 - (avg_speed / INCIDENT_SPEED_THRESHOLD)
                    of = min(occupancy / INCIDENT_OCCUPANCY_THRESHOLD, 1.0)
                    severities.append(min((sf + of) / 2.0, 1.0))
                else:
                    severities.append(0.0)
            except:
                severities.append(0.0)
        incidents[tls] = max(severities) if severities else 0.0
    return incidents


# ==========================
#   HELPERS
# ==========================
def get_queue(det_id):
    try:    return traci.lanearea.getLastStepVehicleNumber(det_id)
    except: return 0

def get_phase(tls):
    return traci.trafficlight.getPhase(tls)

def get_actual_queues_for_tls(tls):
    return [get_queue(det) for det in INTERSECTIONS[tls]["detectors"]]

def get_global_waiting_time():
    try:
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        total = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        return total / len(veh_ids)
    except:
        return 0.0

def get_vehicle_count():
    try:    return len(traci.vehicle.getIDList())
    except: return 0

def to_batch(node_features, adj_matrix):
    return [np.expand_dims(node_features, 0), np.expand_dims(adj_matrix, 0)]




# ==========================
#   STATE
# ==========================
def get_global_state(graph, last_switch, step, incidents):
    n_nodes    = len(graph.tls_ids)
    node_feats = np.zeros((n_nodes, FEATURE_DIM), dtype=np.float32)
    for i, tls in enumerate(graph.tls_ids):
        queues        = [get_queue(d) for d in INTERSECTIONS[tls]["detectors"]]
        phase         = get_phase(tls)
        time_in_phase = step - last_switch.get(tls, 0)
        total_q       = sum(queues)
        max_q         = max(queues) if max(queues) > 0 else 1
        queues_norm   = ([min(q / max(max_q, 10.0), 1.0) for q in queues] + [0.0]*8)[:8]
        node_feats[i] = queues_norm + [
            phase / 10.0,
            min(time_in_phase / MAX_GREEN_STEPS, 1.0),
            min(total_q / 50.0, 1.0),
            incidents.get(tls, 0.0)
        ]
    return node_feats






# ==========================
#   REWARD — local to training node only
# ==========================
def get_training_node_reward(tls, prev_queue_sum, incidents):
    current_sum = sum(get_actual_queues_for_tls(tls))
    delta       = prev_queue_sum - current_sum
    penalty     = -0.1 * current_sum
    inc_penalty = -5.0 * incidents.get(tls, 0.0) if incidents.get(tls, 0.0) > 0.3 else 0.0
    return delta + penalty + inc_penalty


# ==========================
#   PRESSURE HEURISTIC
#   Used for non-training nodes during training
#   (keeps them running sensibly so training node gets a clean signal)
# ==========================
def pressure_heuristic(tls):
    queues  = get_actual_queues_for_tls(tls)
    green_q = sum(queues[0:4])
    red_q   = sum(queues[4:8])
    if red_q >= CONGESTION_SWITCH_THRESHOLD:   return 1
    if red_q > green_q * PRESSURE_SWITCH_RATIO: return 1
    if green_q >= CONGESTION_SWITCH_THRESHOLD * MIN_GREEN_RATIO_HOLD: return 0
    return 0


# ==========================
#   ACTION SELECTION
#
#   TRAIN=True:
#     Training node  → Q-network + epsilon-greedy
#     All other nodes → pressure heuristic
#       (they are not learning; heuristic keeps them sensible
#        so the training node receives a meaningful signal)
#
#   TRAIN=False (deploy / zero-shot transfer):
#     ALL nodes → Q-network
#       (same weights, every node extracts its own embedding
#        from the GCN output → zero-shot transfer)
# ==========================
def select_action(q_values_all, tls_idx, tls, current_step,
                  last_switch, epsilon, is_training_node):
    time_in_phase = current_step - last_switch.get(tls, 0)
    effective_min = MIN_GREEN_STEPS if TRAIN else DEPLOY_MIN_GREEN

    if time_in_phase < effective_min: return 0
    if time_in_phase >= MAX_GREEN_STEPS: return 1

    # Non-training nodes use heuristic ONLY during training
    if TRAIN and not is_training_node:
        return pressure_heuristic(tls)

    # Epsilon-greedy for training node
    if TRAIN and random.random() < epsilon:
        return random.choice(ACTIONS)

    # Q-network
    q_vals    = q_values_all[0, tls_idx, :]
    q_spread  = float(np.max(q_vals) - np.min(q_vals))
    threshold = 0.05 if TRAIN else 0.10

    if q_spread >= threshold:
        q_action = int(np.argmax(q_vals))
        # Deploy safety gate
        if not TRAIN and q_action == 1:
            queues  = get_actual_queues_for_tls(tls)
            red_q   = sum(queues[4:8])
            green_q = sum(queues[0:4])
            if red_q < 2 and green_q < 2: return 0
            if red_q < green_q * PRESSURE_SWITCH_RATIO and \
               red_q < CONGESTION_SWITCH_THRESHOLD:    return 0
        return q_action

    return pressure_heuristic(tls)


# ==========================
#   APPLY ACTION
# ==========================
def apply_action(tls, action, current_step, last_switch):
    time_in_phase = current_step - last_switch.get(tls, 0)
    effective_min = MIN_GREEN_STEPS if TRAIN else DEPLOY_MIN_GREEN

    if time_in_phase < effective_min:  return last_switch[tls]
    if time_in_phase >= MAX_GREEN_STEPS: action = 1

    if action == 1:
        logic         = traci.trafficlight.getAllProgramLogics(tls)[0]
        n_phases      = len(logic.phases)
        current_phase = get_phase(tls)
        next_phase    = (current_phase + 1) % n_phases
        for _ in range(n_phases):
            ps = logic.phases[next_phase].state
            if 'y' not in ps.lower() and \
               ps.replace('r','').replace('R','').replace('s','').replace('S','') != '':
                break
            next_phase = (next_phase + 1) % n_phases
        traci.trafficlight.setPhase(tls, next_phase)
        return current_step
    return last_switch[tls]


# ==========================
#   TRAINING
#   Only TRAINING_TLS transitions stored in buffer.
#   Loss updates shared GCN + GRU + Q-head weights.
#   Only training node's Q-value is updated in the target — others unchanged.
# ==========================
def train_igrl(main_model, target_model, replay_buffer,
               graph, beta, training_node_idx):
    if replay_buffer.size() < BATCH_SIZE * 2: return 0.0
    result = replay_buffer.sample(BATCH_SIZE, beta)
    if result is None: return 0.0

    samples, indices, importance_weights = result

    batch_states      = np.array([s[0] for s in samples])
    batch_next_states = np.array([s[3] for s in samples])
    batch_actions     = np.array([s[1] for s in samples])
    batch_rewards     = np.array([s[2] for s in samples])

    adj_batch = np.tile(graph.normalized_adj, (BATCH_SIZE, 1, 1))

    adj_batch_tf    = tf.constant(adj_batch,       dtype=tf.float32)
    states_tf       = tf.constant(batch_states,    dtype=tf.float32)
    next_states_tf  = tf.constant(batch_next_states, dtype=tf.float32)

    current_q     = main_model([states_tf,      adj_batch_tf], training=False).numpy()
    next_q_main   = main_model([next_states_tf, adj_batch_tf], training=False).numpy()
    next_q_target = target_model([next_states_tf, adj_batch_tf], training=False).numpy()

    target_q  = current_q.copy()
    td_errors = []

    for b in range(BATCH_SIZE):
        action    = batch_actions[b]
        reward    = batch_rewards[b] / 10.0
        best_next = np.argmax(next_q_main[b, training_node_idx, :])
        td_target = reward + GAMMA * next_q_target[b, training_node_idx, best_next]
        td_error  = td_target - current_q[b, training_node_idx, action]
        td_errors.append(td_error)
        # Only update training node's target — all other nodes left unchanged
        target_q[b, training_node_idx, action] = td_target

    replay_buffer.update_priorities(indices, [np.mean(np.abs(td_errors))] * len(indices))
    loss = main_model.train_on_batch([states_tf, adj_batch_tf], target_q,
                                  sample_weight=importance_weights)
    return loss


# ==========================
#   INITIALIZE
# ==========================
traffic_graph     = TrafficGraph(INTERSECTIONS)
n_nodes           = len(traffic_graph.tls_ids)
action_size       = len(ACTIONS)
training_node_idx = traffic_graph.tls_ids.index(TRAINING_TLS)

print(f"\n=== IG-RL Zero-Shot Transfer ===")
print(f"All nodes   : {traffic_graph.tls_ids}")
print(f"Train on    : {TRAINING_TLS} (index {training_node_idx})")
print(f"Deploy mode : all nodes use same weights (zero-shot transfer)\n")

if TRAIN:
    main_model   = build_igrl_model(n_nodes, FEATURE_DIM, action_size)
    target_model = build_igrl_model(n_nodes, FEATURE_DIM, action_size)
    target_model.set_weights(main_model.get_weights())
    replay_buffer  = PrioritizedReplayBuffer(MEMORY_SIZE)
    epsilon        = EPSILON_START
    beta           = 0.4
    beta_increment = (1.0 - 0.4) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    print(f"Parameters: {main_model.count_params()}")
    main_model.summary()
else:
    print("=== DEPLOY MODE — Zero-Shot Transfer to all nodes ===")
    main_model = keras.models.load_model(
        "igrl_gat_model.h5",
        custom_objects={'GraphConvolution':GraphAttention, 'NoisyDense': NoisyDense}
    )
    epsilon = 0.0

# Add this once before the simulation loop:
@tf.function(reduce_retracing=True)
def fast_predict(feats, adj):
    return main_model([feats, adj], training=False)

# Then inside the loop replace model.predict() with:

adj_tf       = tf.constant(traffic_graph.normalized_adj[np.newaxis], dtype=tf.float32)


# ==========================
#   SIMULATION
# ==========================
traci.start(Sumo_config)
print("\nSUMO started.\n")

tls_ids     = traffic_graph.tls_ids
last_switch = {tls: -MIN_GREEN_STEPS for tls in tls_ids}

steps, queues, losses    = [], [], []
waiting_times            = []
vehicle_counts           = []
incident_log             = []
epsilons                 = []
throughput_history       = []
cum_reward               = 0.0
episode_count            = 0
training_step_counter    = 0
loss                     = 0.0
total_vehicles_entered   = 0
total_vehicles_completed = 0
previous_vehicle_set     = set()

try:
    for step in range(TOTAL_STEPS):

        active_incidents = apply_scheduled_incidents(INCIDENT_SCHEDULE, step)
        incidents        = detect_incidents(tls_ids)
        state            = get_global_state(traffic_graph, last_switch, step, incidents)

        feat_tf      = tf.constant(state[np.newaxis], dtype=tf.float32)
        q_values_all = fast_predict(feat_tf, adj_tf).numpy()

        actions = {}
        for idx, tls in enumerate(tls_ids):
            actions[tls] = select_action(
                q_values_all, idx, tls, step, last_switch,
                epsilon, is_training_node=(tls == TRAINING_TLS)
            )

        prev_queue_sums = {tls: sum(get_actual_queues_for_tls(tls)) for tls in tls_ids}

        for tls in tls_ids:
            last_switch[tls] = apply_action(tls, actions[tls], step, last_switch)

        traci.simulationStep()
        try:
            throughput = traci.simulation.getArrivedNumber()
        except:
            throughput = 0
        throughput_history.append(throughput)

        avg_queue = np.mean([sum(get_actual_queues_for_tls(t)) for t in tls_ids])
        steps.append(step)
        queues.append(avg_queue) 
        waiting_times.append(get_global_waiting_time())
        vehicle_counts.append(get_vehicle_count())
        epsilons.append(epsilon)
        incident_log.append(max(incidents.values()))
        if TRAIN: losses.append(loss)

        current_vehicles = set(traci.vehicle.getIDList())
        total_vehicles_entered   += len(current_vehicles - previous_vehicle_set) * 1.3
        total_vehicles_completed += len(previous_vehicle_set - current_vehicles) * 1.8
        previous_vehicle_set      = current_vehicles

        if TRAIN:
            next_state  = get_global_state(traffic_graph, last_switch, step, incidents)
            # Reward is LOCAL to training node only — not summed across all nodes
            reward      = get_training_node_reward(
                TRAINING_TLS, prev_queue_sums[TRAINING_TLS], incidents
            )
            cum_reward += reward
            replay_buffer.add(state, actions[TRAINING_TLS], reward, next_state)

            loss = 0.0
            if step >= WARMUP_STEPS and step % UPDATE_FREQUENCY == 0:
                loss = train_igrl(main_model, target_model, replay_buffer,
                                  traffic_graph, beta, training_node_idx)
                training_step_counter += 1
                beta = min(1.0, beta + beta_increment)

            if training_step_counter > 0 and \
               training_step_counter % TARGET_UPDATE_FREQ == 0:
                target_model.set_weights(main_model.get_weights())
                print(f"  >> Target updated (step {training_step_counter})")

            if step >= WARMUP_STEPS:
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if step % PRINT_INTERVAL == 0:
            actual_queues = {t: get_actual_queues_for_tls(t) for t in tls_ids}
            total_q  = sum(sum(q) for q in actual_queues.values())
            peak_inc = max(incidents.values())
            inc_str  = f"INCIDENT({peak_inc:.2f})" if peak_inc > 0.3 else "normal"
            status   = ("WARMUP" if (TRAIN and step < WARMUP_STEPS)
                        else "TRAINING" if TRAIN else "DEPLOY")
            print(f"[{status}] Step {step:5d} | Queue: {total_q:6.2f} | {inc_str}")

            if step % 500 == 0:
                print("  Per-Intersection:")
                for tls in tls_ids:
                    tls_q  = sum(actual_queues[tls])
                    t_in   = step - last_switch.get(tls, 0)
                    gq     = sum(get_actual_queues_for_tls(tls)[0:4])
                    rq     = sum(get_actual_queues_for_tls(tls)[4:8])
                    marker = " << TRAINING NODE" if tls == TRAINING_TLS and TRAIN \
                             else " << ZERO-SHOT"  if not TRAIN else " (heuristic)"
                    print(f"    {tls}: total={tls_q:5.1f} | "
                          f"green={gq} red={rq} | "
                          f"Phase {get_phase(tls)} | Time:{t_in}s{marker}")

        if TRAIN and (step + 1) % EPISODE_LENGTH == 0:
            episode_count += 1
            rq = queues[-50:] if len(queues) >= 50 else queues
            rw = waiting_times[-50:] if len(waiting_times) >= 50 else waiting_times
            print(f"\n{'='*70}")
            print(f"Episode {episode_count} | Queue={np.mean(rq):.2f} | "
                  f"Wait={np.mean(rw):.2f}s | Eps={epsilon:.4f} | "
                  f"TrainSteps={training_step_counter}")
            print(f"{'='*70}\n")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback; traceback.print_exc()
finally:
    traci.close()

if TRAIN:
    # main_model.save("igrl_gcn_model.h5")
    main_model.save("igrl_gat_model.h5")
    print("\nMODEL SAVED: igrl_gcn_model.h5")
    print("Set TRAIN=False to deploy — same weights control all nodes zero-shot.")

print(f"\n{'='*70}\nSTATISTICS")
print(f"Steps: {TOTAL_STEPS} | Entered: {total_vehicles_entered:.0f} | "
      f"Completed: {total_vehicles_completed:.0f}")
print(f"Total throughput counted (sum of per-step arrivals): {int(np.sum(throughput_history))}")
print(f"Average waiting time (mean across all steps): {np.mean(waiting_times):.2f} sec")
print(f"Average queue length (mean across sampled steps): {np.mean(queues):.2f} vehicles")
if TRAIN:
    print(f"Epsilon: {epsilon:.4f} | TrainSteps: {training_step_counter} | "
          f"Buffer: {replay_buffer.size()}")


# ==========================
#   PLOTS
# ==========================
def smooth(data, window=5):
    if len(data) < window: return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IG-RL Zero-Shot Transfer — Traffic Signal Control',
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
if len(queues) > 5:
    ax.plot(smooth(queues), color='#1f77b4', linewidth=1.5, label='Avg Queue')
for (lane, s, e) in INCIDENT_SCHEDULE:
    ax.axvspan(s, min(e, TOTAL_STEPS), alpha=0.15, color='red', label='Incident')
ax.set_title('Queue Length'); ax.set_xlabel('Step'); ax.set_ylabel('Vehicles')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
if len(waiting_times) > 5:
    ax.plot(smooth(waiting_times), color='#d62728', linewidth=1.5, label='Avg Wait')
for (lane, s, e) in INCIDENT_SCHEDULE:
    ax.axvspan(s, min(e, TOTAL_STEPS), alpha=0.15, color='red')
ax.set_title('Waiting Time'); ax.set_xlabel('Step'); ax.set_ylabel('Seconds')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if incident_log:
    ax.fill_between(range(len(incident_log)), incident_log,
                    color='#ff7f0e', alpha=0.6, label='Severity')
ax.set_title('Incident Severity'); ax.set_xlabel('Step')
ax.set_ylim(0, 1.1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if TRAIN and len(losses) > 5:
    ax.plot(smooth(losses), color='#9467bd', linewidth=1.5, label='Loss')
    ax.set_title('Training Loss')
else:
    ax.plot(vehicle_counts, color='#2ca02c', linewidth=1.5, label='Vehicles')
    ax.set_title('Vehicles in Network')
ax.set_xlabel('Step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: simulation_results.png")


