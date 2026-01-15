CSV_PATH = "..\\csvfiles\\data_all.csv"
CLEANED_CSV = "..\\csvfiles\\data_all_cleaned.csv"
WS_STATE_FILE = "..\\csvfiles\\ws_state.json"
WEIGHTS_STORE_CSV = "..\\csvfiles\\weights_store.csv"
WEIGHTS_HISTORY_CSV = "..\\csvfiles\\weights_history.csv"
TOP_PATHS = "..\\csvfiles\\top_paths.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_BASE_WEIGHT = 0.2
GLOBAL_LAST_TOUCH_DECAY = 0.2
LAST_TOUCH_DECAY_BY_ACTION = {
    "Email": 0.10, "Call": 0.20, "Meeting": 0.29, "Follow Up": 0.05, "1St Appointment": 0.25,
    "2Nd Appointment": 0.27, "Review": 0.10, "Inbound Call": 0.30, "Outbound Call": 0.19,
    "Demo": 0.20, "Discovery": 0.10, "On-Site": 0.30, "UNKNOWN": 0.15
}
MIN_ADJUSTED_WEIGHT = 0.01
MAX_JOURNEY_STEPS = 5
WIN_THRESHOLD = 0.8
LOSS_THRESHOLD = 0.8

ACTION_CANONICAL_MAPPING = {
    "EMAIL": "Email", "E-MAIL": "Email", "EMAILS": "Email",
    "CALL": "Call", "PHONE CALL": "Call",
    "MEETING": "Meeting", "MEET": "Meeting",
    "FOLLOW UP": "Follow Up", "FOLLOW-UP": "Follow Up", "FOLLOWUP": "Follow Up",
    "REVIEW": "Review", "DEMO": "Demo", "SOCIAL": "Social", "OTHER": "Other",
    "NONE": "NONE",

}

NO_OPP_KEYWORDS = {
    "no_opp", "", "None", "null", "Negotiate", "Diagnose",
    "04 - Offer", "1 - Qualificationâ€‹", "03 - Qualified", "8 - Disqualified",
    "Stage 2: Qualified Renewal", "4 - Negotiate", "Implemented", "3 - Design", "5 - Procurement",
    "Stage 8: â‚¬0 Contract Change", "01 - New", "Stage 5: Procurement/Negotiation",
    "SDS - 05 - Negotiating", "2 - Validation", "Access", "Design", "Delivery", "Discovery"
}