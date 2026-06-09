import os

# ================== Project Root ==================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== Data Paths ==================
# Raw data
RAW_STRING = os.path.join(ROOT_DIR, "raw_data/string/9606.protein.links.v12.0 (1).txt")
RAW_EMBEDDINGS = os.path.join(ROOT_DIR, "raw_data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
RAW_INFO = os.path.join(ROOT_DIR, "raw_data/embeddings/9606.protein.info.v12.0.txt.gz")
RAW_SKEMPI = os.path.join(ROOT_DIR, "raw_data/skempi/skempi_v2.xlsx")
RAW_P53_FASTA = os.path.join(ROOT_DIR, "raw_data/p53/p53_sequence.fasta")

# Processed data
DATA_DIR = os.path.join(ROOT_DIR, "data")
PPI_SAMPLE = os.path.join(DATA_DIR, "ppi/ppi_sample_100k_spaced.npz")
SKEMPI_CLEAN = os.path.join(DATA_DIR, "skempi/skempi_clean.csv")
RESIDUE_EMB_DIR = os.path.join(DATA_DIR, "residue_embeddings")
P53_MUTATIONS_DIR = os.path.join(DATA_DIR, "p53_mutations")
ESM2_DATA_DIR = os.path.join(DATA_DIR, "esm2")

# Results
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ================== Model Paths ==================
LOCAL_ENCODER_PROTT5 = os.path.join(ROOT_DIR, "locality_probing/local_encoder/local_encoder.pt")
LOCAL_ENCODER_COSINE = os.path.join(ROOT_DIR, "locality_probing/local_encoder/local_encoder_cosine.pt")
LOCAL_ENCODER_ESM2 = os.path.join(ROOT_DIR, "locality_probing/local_encoder/local_encoder_esm2.pt")
LOCAL_ENCODER_ESM2_V2 = os.path.join(ROOT_DIR, "locality_probing/local_encoder/local_encoder_esm2_v2.pt")
LOCAL_ENCODER_HOLDOUT_P53 = os.path.join(ROOT_DIR, "locality_probing/local_encoder/local_encoder_holdout_p53.pt")

# ================== Model Hyperparameters ==================
# ProtT5 encoder
PROTT5_EMBED_DIM = 1024
PROTT5_WINDOW_LEN = 21
PROTT5_BATCH_SIZE = 16
PROTT5_EPOCHS = 100
PROTT5_LEARNING_RATE = 1e-3

# ESM-2 encoder
ESM2_EMBED_DIM = 640
ESM2_WINDOW_LEN = 61
ESM2_BATCH_SIZE = 32
ESM2_EPOCHS = 150
ESM2_LEARNING_RATE = 1e-3
ESM2_LAMBDA_COS = 0.5

# Deep encoder
DEEP_EMBED_DIM = 64
DEEP_HIDDEN_DIM = 256
DEEP_OUTPUT_DIM = 1024
DEEP_NUM_LAYERS = 4
DEEP_NHEAD = 4
DEEP_MAX_WINDOW_LEN = 41

# ================== Training Settings ==================
DEVICE = "cpu"
TRAIN_TEST_SPLIT = 0.3
VAL_TEST_SPLIT = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 20

# ================== Analysis Settings ==================
# Convergence analysis radii
ANALYSIS_RADII = [2, 5, 10, 15, 20, 30]
MAX_ANALYSIS_WINDOW = 61

# P53 mutations
P53_NOC_MAP = {
    'R175H': 42,
    'G245S': 32,
    'R249S': 30,
    'R282W': 36,
    'Y220C': 32
}

# Amino acid vocabulary
AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
