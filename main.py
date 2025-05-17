import asyncio
import datetime
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import math

from nicegui import app, ui, Client

# --- Constants ---
PLAYERS_PRESET = [
    "Тима", "Алина", "Саня", "Андрей", "Дима Т.", "Эл", "Варя", "Ваня",
    "Петя", "Ася", "Лиза", "Костя", "Даня", "Аня", "Дима М."
]
FORBIDDEN_PAIRS_NAMES = [
    ("Лиза", "Ваня"), ("Лиза", "Ася"), ("Ася", "Ваня"), ("Ваня", "Петя"),
    ("Лиза", "Петя"), ("Андрей", "Саня"), ("Андрей", "Дима Т."),
    ("Дима Т.", "Саня"), ("Дима Т.", "Эл"), ("Эл", "Варя"), ("Даня", "Аня")
]
FORBIDDEN_PAIRS = [frozenset(pair) for pair in FORBIDDEN_PAIRS_NAMES]

STATE_FILE = Path("state.json")
AUTOSAVE_INTERVAL = 30  # seconds
# Tournament end time: 17.05.2025, 21:00 MSK. Assuming computer time is MSK.
# For testing, you might want to adjust this year/date or make it more dynamic.
TOURNAMENT_END_DATETIME = datetime.datetime(2025, 5, 17, 4, 50, 0)
# TOURNAMENT_END_DATETIME = datetime.datetime.now() + datetime.timedelta(minutes=5) # For quick testing

MAX_TABLES = 4

# --- Data Models ---
class Player:
    def __init__(self, name: str):
        self.name: str = name
        self.present: bool = True 
        self.wins: int = 0
        self.losses: int = 0
        self.break_first_count: int = 0
        self.played_with: Set[str] = set() # All opponents
        self.opponents_defeated: Set[str] = set() # For SB score
        self.win_durations: List[float] = [] # Durations of matches won (seconds)
        self.extra_games_played: int = 0
        self.last_played_timestamp: float = 0.0 # time.time() when their last game ENDED
        self.current_rank: int = 0

    def average_win_time(self) -> Optional[float]:
        if not self.win_durations:
            return None
        return sum(self.win_durations) / len(self.win_durations)

    def sonnenborn_berger_score(self, all_players_map: Dict[str, 'Player']) -> float:
        score = 0.0
        for opponent_name in self.opponents_defeated:
            if opponent_name in all_players_map:
                score += all_players_map[opponent_name].wins
        return score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "present": self.present, "wins": self.wins, "losses": self.losses,
            "break_first_count": self.break_first_count, "played_with": list(self.played_with),
            "opponents_defeated": list(self.opponents_defeated), "win_durations": self.win_durations,
            "extra_games_played": self.extra_games_played, 
            "last_played_timestamp": self.last_played_timestamp,
            "current_rank": self.current_rank
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        player = cls(data["name"])
        player.present = data.get("present", True)
        player.wins = data.get("wins", 0)
        player.losses = data.get("losses", 0)
        player.break_first_count = data.get("break_first_count", 0)
        player.played_with = set(data.get("played_with", []))
        player.opponents_defeated = set(data.get("opponents_defeated", []))
        player.win_durations = data.get("win_durations", [])
        player.extra_games_played = data.get("extra_games_played", 0)
        player.last_played_timestamp = data.get("last_played_timestamp", 0.0)
        player.current_rank = data.get("current_rank", 0)
        return player

class Match:
    def __init__(self, p_red: str, p_blue: str, table_name: str, table_idx: int):
        self.p_red: str = p_red
        self.p_blue: str = p_blue
        self.table_name: str = table_name
        self.table_idx: int = table_idx
        self.start_time: Optional[datetime.datetime] = None # Actual datetime object
        self.end_time: Optional[datetime.datetime] = None   # Actual datetime object
        self.winner: Optional[str] = None
        self.match_timer_start_monotonic: Optional[float] = None # time.monotonic() for elapsed timer

    def duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def elapsed_seconds(self) -> float:
        if self.match_timer_start_monotonic is None:
            return 0.0
        return time.monotonic() - self.match_timer_start_monotonic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_red": self.p_red, "p_blue": self.p_blue, "table_name": self.table_name,
            "table_idx": self.table_idx,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "winner": self.winner,
            "match_timer_start_monotonic": self.match_timer_start_monotonic # Will not be restored perfectly
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Match':
        match = cls(data["p_red"], data["p_blue"], data["table_name"], data["table_idx"])
        match.start_time = datetime.datetime.fromisoformat(data["start_time"]) if data["start_time"] else None
        match.end_time = datetime.datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        match.winner = data["winner"]
        # match_timer_start_monotonic is tricky to restore. If match is active on load, timer might be off.
        # For simplicity, if a match was active, its timer restarts from 0 upon loading.
        # Or, if start_time exists, we can estimate:
        if match.start_time and not match.end_time: # Active match
             elapsed_on_save = data.get("match_timer_start_monotonic") # This was a time.monotonic() value, not duration
             # A better way: store elapsed time directly, or re-calculate from start_time.
             # For now, let's re-set it as if it just started to avoid complexity.
             # Or, calculate from persisted start_time.
             now_monotonic = time.monotonic()
             time_since_start = (datetime.datetime.now() - match.start_time).total_seconds()
             match.match_timer_start_monotonic = now_monotonic - time_since_start
        return match

class GameState:
    def __init__(self):
        self.players: Dict[str, Player] = {name: Player(name) for name in PLAYERS_PRESET}
        self.table_names: List[str] = [f"Стол {i+1}" for i in range(MAX_TABLES)]
        self.active_matches: Dict[int, Match] = {} # table_idx -> Match
        self.match_queue: List[Tuple[str, Optional[str]]] = []
        self.finished_matches: List[Match] = []
        self.round_idx: int = 0
        self.current_round_start_time: Optional[datetime.datetime] = None
        self.tournament_started: bool = False
        self.tournament_finish_screen_shown: bool = False
        self.played_pairs: Set[frozenset[str]] = set()
        self.last_save_time: float = time.time()
        self.initial_ranks: Dict[str, int] = {} # For "best comeback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "players": {name: p.to_dict() for name, p in self.players.items()},
            "table_names": self.table_names,
            "active_matches": {idx: m.to_dict() for idx, m in self.active_matches.items()},
            "match_queue": self.match_queue,
            "finished_matches": [m.to_dict() for m in self.finished_matches],
            "round_idx": self.round_idx,
            "current_round_start_time": self.current_round_start_time.isoformat() if self.current_round_start_time else None,
            "tournament_started": self.tournament_started,
            "tournament_finish_screen_shown": self.tournament_finish_screen_shown,
            "played_pairs": [list(pair) for pair in self.played_pairs],
            "initial_ranks": self.initial_ranks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        gs = cls()
        gs.players = {name: Player.from_dict(p_data) for name, p_data in data.get("players", {}).items()}
        # Ensure all preset players are there, adding defaults if any are missing from save
        for p_name in PLAYERS_PRESET:
            if p_name not in gs.players:
                gs.players[p_name] = Player(p_name)

        gs.table_names = data.get("table_names", [f"Стол {i+1}" for i in range(MAX_TABLES)])
        gs.active_matches = {
            int(idx_str): Match.from_dict(m_data)
            for idx_str, m_data in data.get("active_matches", {}).items()
        }
        gs.match_queue = [tuple(pair) for pair in data.get("match_queue", [])] # Ensure tuple format
        gs.finished_matches = [Match.from_dict(m_data) for m_data in data.get("finished_matches", [])]
        gs.round_idx = data.get("round_idx", 0)
        current_round_start_time_str = data.get("current_round_start_time")
        gs.current_round_start_time = datetime.datetime.fromisoformat(current_round_start_time_str) if current_round_start_time_str else None
        gs.tournament_started = data.get("tournament_started", False)
        gs.tournament_finish_screen_shown = data.get("tournament_finish_screen_shown", False)
        gs.played_pairs = set(frozenset(pair_list) for pair_list in data.get("played_pairs", []))
        gs.initial_ranks = data.get("initial_ranks", {})
        return gs

gs = GameState() # Global state variable

# --- UI References ---
# These will be populated by UI building functions
ui_container: Optional[ui.column] = None
top_bar_round_label: Optional[ui.label] = None
top_bar_round_time_label: Optional[ui.label] = None
top_bar_tournament_time_label: Optional[ui.label] = None
table_cards_display: Dict[int, Dict[str, Any]] = {} # table_idx -> {'card', 'p_red_btn', 'p_blue_btn', 'timer_label', 'status_label'}
scoreboard_container: Optional[ui.html] = None # Using HTML for more table control

# --- State Persistence ---
def save_state_sync():
    global gs
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(gs.to_dict(), f, indent=2, ensure_ascii=False)
        gs.last_save_time = time.time()
        # print(f"State saved: {datetime.datetime.now()}")
    except Exception as e:
        print(f"Error saving state: {e}")

async def save_state(): # Async wrapper for use in NiceGUI
    save_state_sync()

def load_state_sync():
    global gs
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            gs = GameState.from_dict(data)
            # print("State loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading state: {e}. Starting fresh.")
            gs = GameState() # Reset
            return False
    return False

# --- Tournament Logic ---
def get_present_players() -> List[Player]:
    return [p for p in gs.players.values() if p.present]

def get_present_player_names() -> List[str]:
    return [p.name for p in gs.players.values() if p.present]

def generate_initial_pairings() -> List[Tuple[str, Optional[str]]]:
    present_names = get_present_player_names()
    if len(present_names) < 2:
        return []

    best_shuffled_list = list(present_names)
    min_forbidden_count = float('inf')

    # Try to find a shuffle that minimizes forbidden pairs
    for _ in range(200): # Increased iterations for better chance
        shuffled_list = random.sample(present_names, len(present_names))
        current_forbidden_count = 0
        for i in range(0, len(shuffled_list) - 1 if len(shuffled_list) % 2 == 0 else len(shuffled_list) - 2, 2):
            p1, p2 = shuffled_list[i], shuffled_list[i+1]
            if frozenset({p1, p2}) in FORBIDDEN_PAIRS:
                current_forbidden_count += 1
        
        if current_forbidden_count < min_forbidden_count:
            min_forbidden_count = current_forbidden_count
            best_shuffled_list = list(shuffled_list)
        
        if min_forbidden_count == 0:
            break
    
    pairings = []
    num_to_pair = len(best_shuffled_list)
    
    # Create pairs from the best shuffle
    paired_indices = set()
    for i in range(num_to_pair):
        if i in paired_indices:
            continue
        p1_name = best_shuffled_list[i]
        
        if i + 1 < num_to_pair : # Check if there's at least one more player to pair with
            p2_name = best_shuffled_list[i+1]
            pairings.append((p1_name, p2_name))
            paired_indices.add(i)
            paired_indices.add(i+1)
        else: # Last player, becomes a floater
            pairings.append((p1_name, None))
            paired_indices.add(i)
            break # Should be the last player

    # print(f"Initial pairings ({min_forbidden_count=}): {pairings}")
    return pairings

def generate_swiss_pairings() -> List[Tuple[str, Optional[str]]]:
    present_players_obj = get_present_players()

    def sort_key(p: Player):
        awt = p.average_win_time()
        return (
            -p.wins,
            awt if awt is not None else float('inf'),
            -p.sonnenborn_berger_score(gs.players)
        )
    
    present_players_obj.sort(key=sort_key)
    
    sorted_player_names = [p.name for p in present_players_obj]
    
    new_pairings = []
    available_players = list(sorted_player_names)
    
    while len(available_players) >= 1: # Changed from >=2 to handle last player correctly
        p1_name = available_players.pop(0)
        found_partner = False
        if not available_players: # p1_name is the last one
            new_pairings.append((p1_name, None))
            break

        # Try to find a partner for p1_name from the rest of available_players
        for i in range(len(available_players)):
            p2_name = available_players[i]
            if frozenset({p1_name, p2_name}) not in gs.played_pairs:
                new_pairings.append((p1_name, p2_name))
                available_players.pop(i) # p2_name is now paired
                found_partner = True
                break
        
        if not found_partner:
            # If no non-repeat partner found, must pick one (least recent repeat - advanced)
            # For now, pick the first available one if repeats are absolutely necessary
            # Or, this player becomes a (player, None) for floater logic.
            # For a more robust Swiss, if no valid partner, p1 might "float down" or pair with force.
            # Simplified: if no ideal partner, becomes floater.
            # print(f"Player {p1_name} could not find a non-repeat partner. Adding as floater candidate.")
            new_pairings.append((p1_name, None)) 
            # Note: this could create multiple (X, None) if pairing is hard. Floater logic will handle.

    # print(f"Swiss pairings: {new_pairings}")
    return new_pairings

def choose_floater_for(player_a_name: str) -> Optional[str]:
    eligible_partners = []
    player_a_obj = gs.players[player_a_name]

    active_player_names = {m.p_red for m in gs.active_matches.values()} | \
                          {m.p_blue for m in gs.active_matches.values()}

    for p_name, p_obj in gs.players.items():
        if not p_obj.present or p_name == player_a_name or p_name in active_player_names:
            continue
        
        # Avoid pairing if they just played (unless no other option)
        # The main constraint is `gs.played_pairs` for strict no-repeat.
        # Here, we are *forced* to pick someone.
        # Priority: 1. Fewest extra games, 2. Longest wait time (lowest last_played_timestamp)
        eligible_partners.append(p_obj)

    if not eligible_partners:
        return None

    eligible_partners.sort(key=lambda p: (p.extra_games_played, p.last_played_timestamp))
    
    # Try to find one who hasn't played player_a_name IF POSSIBLE
    chosen_partner = None
    for partner_obj in eligible_partners:
        if frozenset({player_a_name, partner_obj.name}) not in gs.played_pairs:
            chosen_partner = partner_obj
            break
    
    if not chosen_partner and eligible_partners: # All eligible partners are repeats
        # Fallback: pick the best from eligible_partners (least extra games, longest wait)
        # This is "least recent repeat" implicitly if timestamps are diverse.
        chosen_partner = eligible_partners[0]
        # print(f"Floater for {player_a_name}: chosen {chosen_partner.name} (repeat pairing necessary)")
    elif not chosen_partner: # No eligible partners at all
        return None
        
    # print(f"Floater for {player_a_name}: chosen {chosen_partner.name}")
    gs.players[chosen_partner.name].extra_games_played += 1
    gs.players[player_a_name].extra_games_played += 1 # The one needing a floater also plays an "extra" game
    return chosen_partner.name

def fair_break(pA_name: str, pB_name: str) -> Tuple[str, str]:
    pA = gs.players[pA_name]
    pB = gs.players[pB_name]

    if pA.break_first_count < pB.break_first_count:
        red, blue = pA_name, pB_name
    elif pB.break_first_count < pA.break_first_count:
        red, blue = pB_name, pA_name
    else:
        players_list = [pA_name, pB_name]
        random.shuffle(players_list)
        red, blue = players_list[0], players_list[1]
    
    gs.players[red].break_first_count += 1
    return red, blue

async def attempt_to_seat_next_match(table_idx: int):
    global gs
    if gs.active_matches.get(table_idx): # Table is busy
        # print(f"DEBUG attempt_to_seat_next_match: Table {table_idx} is busy.")
        return
    
    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    if tournament_time_over and not any(gs.active_matches.values()):
         await show_finish_screen_if_needed()
         return
    if tournament_time_over and gs.match_queue:
        update_table_card_ui_for_idle(table_idx, "Время вышло, стол свободен")
        return

    if not gs.match_queue:
        # Queue is empty. Check if we should generate a new round.
        if not gs.active_matches: # All tables free, AND queue empty. This is the main condition for new round.
            if gs.round_idx == 0 : 
                 # This case should be handled by initial pairings already in queue.
                 # If somehow reached here at round 0 with empty queue and no active matches, it's unusual.
                 print("WARN: attempt_to_seat_next_match called at round 0 with empty queue and no active matches.")
                 pass 
            # else: # Subsequent rounds (or round 0 if initial setup didn't populate queue for some reason)
            print(f"DEBUG: Table {table_idx} finds queue empty and all tables free. Attempting to generate new round.")
            new_round_pairings = generate_swiss_pairings()
            
            # Check if any actual pairs (not (player, None) that couldn't be resolved) were made
            # valid_new_pairs_generated = any(p_tuple for p_tuple in new_round_pairings if p_tuple[1] is not None)

            gs.match_queue.extend(new_round_pairings) # Extend queue with what was generated

            if new_round_pairings: # If any pairings (even (X, None)) were returned
                gs.round_idx += 1
                gs.current_round_start_time = datetime.datetime.now()
                print(f"Advanced to Round {gs.round_idx}. New pairs added to queue: {len(new_round_pairings)}")
                await save_state() # Save state after advancing round and getting new pairs

                # --- MODIFICATION START ---
                # Since new pairs are available (or at least new (Player, None) entries),
                # try to seat them on ALL idle tables.
                # The current call to attempt_to_seat_next_match(table_idx) will continue
                # and try to seat on 'table_idx'. This loop handles other tables.
                for i in range(MAX_TABLES):
                    if i not in gs.active_matches: # If table 'i' is currently idle
                        # We don't need to check if i == table_idx, as the current function instance
                        # will attempt to seat table_idx. If another task for table_idx runs,
                        # it will find table_idx busy (if seated by this instance) or queue changed.
                        print(f"DEBUG: New round {gs.round_idx} generated. Proactively trying to seat on idle table {i} (triggered by table {table_idx})")
                        asyncio.create_task(attempt_to_seat_next_match(i))
                # --- MODIFICATION END ---
            else:
                print(f"No new pairings generated for round {gs.round_idx + 1}.")
        
        # After attempting to generate, check queue again. If still empty, then table waits.
        if not gs.match_queue: 
            # This message is crucial. If it persists, it means no pairs could be formed or queue is truly empty.
            update_table_card_ui_for_idle(table_idx, "Ожидаем следующий раунд (нет пар)") 
            return

    # --- From here, assume gs.match_queue MIGHT have entries ---

    # Try to get a pair from the queue
    # (The rest of your existing logic for popping from queue, handling floaters, seating match)
    # ... (your existing code for best_pair_idx, p1_raw, p2_raw_opt, etc.)
    best_pair_idx = -1
    is_floater_pair = False # Flag to know if we popped a (P, None) pair

    # Search for a P1,P2 pair first
    for i, pair_tuple in enumerate(gs.match_queue):
        p1_name, p2_name_opt = pair_tuple
        if p1_name and p2_name_opt: # This is a complete pair (P1, P2)
            if frozenset({p1_name, p2_name_opt}) in gs.played_pairs:
                # This pair should ideally be filtered out during generation or floater resolution
                # print(f"DEBUG: Skipping already played pair {pair_tuple} from queue for table {table_idx}.")
                continue
            best_pair_idx = i
            is_floater_pair = False
            break # Found a P1,P2 pair

    # If no P1,P2 pair, look for a P1,None floater candidate (if any were queued)
    if best_pair_idx == -1:
        for i, pair_tuple in enumerate(gs.match_queue):
            p1_name, p2_name_opt = pair_tuple
            if p1_name and p2_name_opt is None: # This is a (Player, None) floater situation
                best_pair_idx = i
                is_floater_pair = True
                break # Found a P1,None pair

    if best_pair_idx == -1:
        update_table_card_ui_for_idle(table_idx, "Нет доступных пар в очереди")
        # print(f"DEBUG: No suitable pair found in queue for table {table_idx}.")
        return

    p1_raw, p2_raw_opt = gs.match_queue.pop(best_pair_idx)
    final_p1_name: str
    final_p2_name: Optional[str]

    if p2_raw_opt is None: # Floater case: (p1_raw, None)
        final_p1_name = p1_raw
        print(f"DEBUG: Handling floater: {final_p1_name} needs partner for table {table_idx}.")
        final_p2_name = choose_floater_for(final_p1_name)
        if final_p2_name is None:
            print(f"DEBUG: Could not find floater partner for {final_p1_name}. Re-queuing {final_p1_name}.")
            gs.match_queue.append((final_p1_name, None)) # Add back
            update_table_card_ui_for_idle(table_idx, f"{final_p1_name} ожидает партнера")
            # Potentially try to seat another type of pair if one was skipped for this floater.
            # For simplicity, just return and let another table or timer trigger next attempt.
            return
    else: # Normal pair (p1_raw, p2_raw_opt)
        final_p1_name = p1_raw
        final_p2_name = p2_raw_opt
        
    current_match_players_set = frozenset({final_p1_name, final_p2_name})
    if current_match_players_set in gs.played_pairs:
        print(f"ERROR: Attempting to seat {final_p1_name} vs {final_p2_name} (already played) on table {table_idx}. Re-queuing original.")
        # Re-add original pair to queue based on what was popped
        if is_floater_pair: # Was (p1_raw, None), but floater logic resulted in a repeat
            gs.match_queue.append((p1_raw, None)) # Re-queue the (P, None)
        else: # Was (p1_raw, p2_raw_opt)
             gs.match_queue.append((p1_raw, p2_raw_opt))
        # To prevent table stall, try to seat another distinct pair if available for THIS table
        # This could be complex; for now, this table will show idle after this error.
        # Another table might pick up a valid pair, or a timer will re-trigger.
        update_table_card_ui_for_idle(table_idx, "Ошибка: Повторная пара")
        # A more robust solution might be to call attempt_to_seat_next_match(table_idx) again
        # but that risks infinite loops if the queue only contains bad pairs.
        # asyncio.create_task(attempt_to_seat_next_match(table_idx)) # Use with caution
        return

    p_red_name, p_blue_name = fair_break(final_p1_name, final_p2_name)
    table_display_name = gs.table_names[table_idx] # Use table_display_name for clarity
    match = Match(p_red=p_red_name, p_blue=p_blue_name, table_name=table_display_name, table_idx=table_idx)
    match.start_time = datetime.datetime.now()
    match.match_timer_start_monotonic = time.monotonic()
    
    gs.active_matches[table_idx] = match
    gs.players[final_p1_name].played_with.add(final_p2_name)
    gs.players[final_p2_name].played_with.add(final_p1_name)
    gs.played_pairs.add(frozenset({final_p1_name, final_p2_name}))

    print(f"Seating match on {table_display_name} (Idx {table_idx}): {p_red_name} (Red) vs {p_blue_name} (Blue)")

    update_table_card_ui_for_match(table_idx, match)
    await save_state()

async def handle_match_result(table_idx: int, winner_name: str):
    global gs
    if table_idx not in gs.active_matches:
        # print(f"Error: No active match on table index {table_idx} to finish.")
        return

    match = gs.active_matches.pop(table_idx)
    match.end_time = datetime.datetime.now()
    match.winner = winner_name
    
    gs.finished_matches.append(match)

    # Update player stats
    winner_obj = gs.players[winner_name]
    loser_name = match.p_red if match.p_blue == winner_name else match.p_blue
    loser_obj = gs.players[loser_name]

    winner_obj.wins += 1
    winner_obj.opponents_defeated.add(loser_name)
    duration_sec = match.duration_seconds()
    if duration_sec is not None:
        winner_obj.win_durations.append(duration_sec)
    
    loser_obj.losses += 1
    
    timestamp_now = time.time()
    winner_obj.last_played_timestamp = timestamp_now
    loser_obj.last_played_timestamp = timestamp_now

    # print(f"Match on {match.table_name} finished. Winner: {winner_name}")

    update_scoreboard_display()
    clear_table_card_ui(table_idx) # Set to "Ожидаем..."
    await save_state() # Save before seating next, critical for recovery

    # Try to seat the next pair on this now-free table
    await attempt_to_seat_next_match(table_idx)

    # Check if tournament should end
    await show_finish_screen_if_needed()
    
# --- UI Rendering & Updates ---
def format_duration(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"

def update_top_bar_timers():
    if not gs.tournament_started or gs.tournament_finish_screen_shown:
        return

    if top_bar_round_label:
        top_bar_round_label.set_text(f"Раунд {gs.round_idx if gs.round_idx > 0 else '1 (Начальный)'}")

    if top_bar_round_time_label and gs.current_round_start_time:
        elapsed_seconds = (datetime.datetime.now() - gs.current_round_start_time).total_seconds()
        top_bar_round_time_label.set_text(f"⏱ Раунда: {format_duration(elapsed_seconds)}")
    elif top_bar_round_time_label:
         top_bar_round_time_label.set_text("⏱ Раунда: --:--")


    if top_bar_tournament_time_label:
        now = datetime.datetime.now()
        if now < TOURNAMENT_END_DATETIME:
            remaining_time = TOURNAMENT_END_DATETIME - now
            # h, rem = divmod(remaining_time.total_seconds(), 3600)
            # m, s = divmod(rem, 60)
            # top_bar_tournament_time_label.set_text(f"До конца: {int(h):02d}:{int(m):02d}:{int(s):02d}")
            top_bar_tournament_time_label.set_text(f"До конца: {str(remaining_time).split('.')[0]}") # Simpler H:MM:SS
            top_bar_tournament_time_label.style(remove="color: red;")
        else:
            over_time = now - TOURNAMENT_END_DATETIME
            # h, rem = divmod(over_time.total_seconds(), 3600)
            # m, s = divmod(rem, 60)
            # top_bar_tournament_time_label.set_text(f"Овертайм: -{int(h):02d}:{int(m):02d}:{int(s):02d}")
            top_bar_tournament_time_label.set_text(f"Овертайм: -{str(over_time).split('.')[0]}")
            top_bar_tournament_time_label.style(add="color: red;")

def update_match_timers():
    if not gs.tournament_started or gs.tournament_finish_screen_shown:
        return
    for table_idx, match in gs.active_matches.items():
        if table_idx in table_cards_display and match.match_timer_start_monotonic is not None:
            elapsed = match.elapsed_seconds()
            table_cards_display[table_idx]['timer_label'].set_text(format_duration(elapsed))

def update_table_card_ui_for_match(table_idx: int, match: Match):
    card_elements = table_cards_display.get(table_idx)
    if not card_elements: 
        # print(f"DEBUG: update_table_card_ui_for_match: No card_elements found for table_idx {table_idx}.")
        return

    p_red_btn_el = card_elements.get('p_red_btn')
    p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label')
    timer_label_el = card_elements.get('timer_label')
    card_el = card_elements.get('card')

    if not all([p_red_btn_el, p_blue_btn_el, status_label_el, timer_label_el, card_el]):
        # print(f"DEBUG: update_table_card_ui_for_match: One or more UI elements are None for table_idx {table_idx}.")
        return

    p_red_btn_el.set_text(match.p_red)
    p_red_btn_el.props('disable=false')

    p_blue_btn_el.set_text(match.p_blue)
    p_blue_btn_el.props('disable=false')

    status_label_el.set_text(match.table_name)
    status_label_el.classes(remove='text-grey')

    timer_label_el.set_text("00:00") # Initial timer display for an active match
    card_el.classes(remove='table-card-idle')

def clear_table_card_ui(table_idx: int):
    update_table_card_ui_for_idle(table_idx, "Ожидаем пару...")

def update_table_card_ui_for_idle(table_idx: int, message: str):
    card_elements = table_cards_display.get(table_idx)
    if not card_elements:
        # print(f"DEBUG: update_table_card_ui_for_idle: No card_elements found for table_idx {table_idx}.")
        return

    p_red_btn_el = card_elements.get('p_red_btn')
    p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label')
    timer_label_el = card_elements.get('timer_label')
    card_el = card_elements.get('card')

    if not all([p_red_btn_el, p_blue_btn_el, status_label_el, timer_label_el, card_el]):
        # print(f"DEBUG: update_table_card_ui_for_idle: One or more UI elements are None for table_idx {table_idx}.")
        # print(f"  p_red_btn: {'Exists' if p_red_btn_el else 'None'}, p_blue_btn: {'Exists' if p_blue_btn_el else 'None'}, ...etc")
        return

    p_red_btn_el.set_text("Игрок 1")
    p_red_btn_el.props('disable=true') # For idle, buttons are disabled

    p_blue_btn_el.set_text("Игрок 2")
    p_blue_btn_el.props('disable=true') # For idle, buttons are disabled

    status_label_el.set_text(message)
    status_label_el.classes(add='text-grey')

    timer_label_el.set_text("--:--")
    card_el.classes(add='table-card-idle')

def update_scoreboard_display():
    if not scoreboard_container or not gs.tournament_started:
        return

    present_players_list = get_present_players()
    
    # Sort players for ranking
    def sort_key(p: Player):
        awt = p.average_win_time()
        sb_score = p.sonnenborn_berger_score(gs.players) # Pass full map for SB calc
        return (
            -p.wins,  # Primary: Wins (desc)
            awt if awt is not None else float('inf'), # Secondary: Avg Win Time (asc)
            -sb_score # Tertiary: SB Score (desc)
        )
    
    present_players_list.sort(key=sort_key)

    # Assign ranks
    # rank_counter = 0
    # last_score_tuple = (None, None, None)
    # for i, p_obj in enumerate(present_players_list):
    #     current_score_tuple = (p_obj.wins, p_obj.average_win_time(), p_obj.sonnenborn_berger_score(gs.players))
    #     if current_score_tuple != last_score_tuple:
    #         rank_counter = i + 1
    #     p_obj.current_rank = rank_counter
    #     last_score_tuple = current_score_tuple
    
    # Simplified ranking: rank is just index + 1 after sort if no ties displayed explicitly
    # Proper rank assignment considering ties:
    ranks = {}
    for i, p_obj in enumerate(present_players_list):
        p_obj.current_rank = i + 1 # Temporary, will be adjusted for ties
        if i == 0:
            ranks[p_obj.name] = 1
        else:
            prev_p_obj = present_players_list[i-1]
            # Compare with previous player based on sort_key logic
            # If p_obj is tied with prev_p_obj, they get same rank as prev_p_obj
            # Otherwise, p_obj gets i+1 rank
            key_curr = sort_key(p_obj)
            key_prev = sort_key(prev_p_obj)
            if key_curr == key_prev:
                ranks[p_obj.name] = ranks[prev_p_obj.name]
            else:
                ranks[p_obj.name] = i + 1
        p_obj.current_rank = ranks[p_obj.name]


    # Build HTML table for scoreboard
    html = "<table class='score-table'><thead><tr><th>#</th><th>Имя</th><th>Побед</th><th>Ср. время</th><th>СБ</th></tr></thead><tbody>"
    if not present_players_list:
        html += "<tr><td colspan='5' style='text-align:center;'>Нет игроков</td></tr>"
    else:
        for p_obj in present_players_list:
            avg_win_time_str = format_duration(p_obj.average_win_time()) if p_obj.average_win_time() is not None else "-"
            sb_str = f"{p_obj.sonnenborn_berger_score(gs.players):.2f}"
            html += f"<tr><td>{p_obj.current_rank}</td><td>{p_obj.name}</td><td>{p_obj.wins}</td><td>{avg_win_time_str}</td><td>{sb_str}</td></tr>"
    html += "</tbody></table>"
    
    scoreboard_container.set_content(html)


async def show_confirmation_dialog(table_idx: int, player_name: str):
    print(f"DEBUG: Attempting to show dialog. Table Idx: {table_idx}, Player: {player_name}")
    try:
        # Ensure table_idx is valid for gs.table_names before using it.
        table_display_name = f"Стол {table_idx + 1}" # Default if not found
        if 0 <= table_idx < len(gs.table_names):
            table_display_name = gs.table_names[table_idx]
        else:
            print(f"WARN: table_idx {table_idx} is out of bounds for gs.table_names (len: {len(gs.table_names)})")

        with ui.dialog() as dialog, ui.card(): # This is where the error occurs
            ui.label(f"Победил {player_name} на столе {table_display_name}?")
            with ui.row().classes("justify-end w-full"): # Added classes for button layout
                ui.button("Нет", on_click=lambda: dialog.submit(False), color='negative')
                ui.button("Да", on_click=lambda: dialog.submit(True), color='positive')
        
        print(f"DEBUG: Dialog elements created for {player_name}. Awaiting result...")
        result = await dialog
        print(f"DEBUG: Dialog for {player_name} submitted with result: {result}")
        
        if result:
            await handle_match_result(table_idx, player_name)
        else:
            # Optionally do something if "Нет" is clicked, e.g., log it
            print(f"INFO: Match result recording cancelled for {player_name} on table {table_idx}.")

    except RuntimeError as e:
        print(f"FATAL ERROR in show_confirmation_dialog creating UI: {e}")
        import traceback
        traceback.print_exc()
        # Optionally, show an on-page notification if possible, though dialog creation itself failed
        ui.notify(f"Ошибка отображения диалога: {e}", type='negative', multi_line=True, close_button=True)
        # Do not re-raise if you want the app to try and continue, 
        # but the user won't get the dialog.
    except Exception as e:
        print(f"UNEXPECTED ERROR in show_confirmation_dialog: {e}")
        import traceback
        traceback.print_exc()
        ui.notify(f"Неожиданная ошибка: {e}", type='negative', multi_line=True, close_button=True)

async def show_finish_screen_if_needed():
    global gs, ui_container
    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    no_active_matches = not gs.active_matches
    
    if tournament_time_over and no_active_matches and gs.tournament_started and not gs.tournament_finish_screen_shown:
        gs.tournament_finish_screen_shown = True # Prevent re-entry
        await save_state() # Save final state before showing podium

        if ui_container is None: return # Should not happen

        ui_container.clear()
        with ui_container:
            ui.label("Турнир Завершен!").classes("text-h3 self-center q-my-md")
            
            # Calculate final rankings again
            present_players_list = get_present_players()
            def sort_key(p: Player):
                awt = p.average_win_time()
                sb_score = p.sonnenborn_berger_score(gs.players)
                return (-p.wins, awt if awt is not None else float('inf'), -sb_score)
            present_players_list.sort(key=sort_key)

            # Assign final ranks properly
            final_ranks_map = {}
            if present_players_list:
                for i, p_obj in enumerate(present_players_list):
                    if i == 0:
                        final_ranks_map[p_obj.name] = 1
                    else:
                        prev_p_obj = present_players_list[i-1]
                        key_curr = sort_key(p_obj)
                        key_prev = sort_key(prev_p_obj)
                        if key_curr == key_prev:
                            final_ranks_map[p_obj.name] = final_ranks_map[prev_p_obj.name]
                        else:
                            final_ranks_map[p_obj.name] = i + 1
                    p_obj.current_rank = final_ranks_map[p_obj.name]


            # Podium
            with ui.row().classes("justify-center q-gutter-md q-my-lg"):
                if len(present_players_list) >= 1:
                    p1 = present_players_list[0]
                    with ui.card().classes("podium-card items-center"):
                        ui.label("🥇").classes("podium-1")
                        ui.label(f"{p1.name}").classes("text-h5")
                        ui.label(f"{p1.wins} побед").classes("text-caption")
                if len(present_players_list) >= 2:
                    # Find actual second place (could be multiple tied for 1st)
                    p2 = next((p for p in present_players_list if final_ranks_map.get(p.name) == 2), None)
                    if not p2 and final_ranks_map.get(present_players_list[0].name) == 1 and len(present_players_list) > 1 and final_ranks_map.get(present_players_list[1].name) != 1: # If P1 is unique 1st
                        p2 = present_players_list[1] # The next unique ranked player
                    
                    if p2 :
                        with ui.card().classes("podium-card items-center"):
                            ui.label("🥈").classes("podium-2")
                            ui.label(f"{p2.name}").classes("text-h6")
                            ui.label(f"{p2.wins} побед").classes("text-caption")
                
                if len(present_players_list) >= 3:
                    p3 = next((p for p in present_players_list if final_ranks_map.get(p.name) == 3), None)
                    # Complex tie logic for 3rd:
                    # If 2 unique people are 1st and 2nd, person at index 2 might be 3rd.
                    # If multiple people tied for 1st or 2nd, finding distinct 3rd is harder.
                    # Simplified: take the 3rd distinct rank if available.
                    if not p3: # Try to find the player at the 3rd rank position
                        rank_1_players = [p for p in present_players_list if final_ranks_map.get(p.name) == 1]
                        rank_2_players = [p for p in present_players_list if final_ranks_map.get(p.name) == 2]
                        
                        current_idx_for_3rd = len(rank_1_players) + len(rank_2_players)
                        if current_idx_for_3rd < len(present_players_list):
                             p3_candidate = present_players_list[current_idx_for_3rd]
                             if final_ranks_map.get(p3_candidate.name) == 3 : # Or check if its rank is indeed 3rd distinct
                                p3 = p3_candidate
                    
                    if p3:
                        with ui.card().classes("podium-card items-center"):
                            ui.label("🥉").classes("podium-3")
                            ui.label(f"{p3.name}").classes("text-subtitle1")
                            ui.label(f"{p3.wins} побед").classes("text-caption")
            
            # Statistics
            with ui.card().classes("q-my-md"):
                ui.label("Интересная статистика:").classes("text-h6")
                if gs.finished_matches:
                    # Fastest Match
                    fastest_match = min(gs.finished_matches, key=lambda m: m.duration_seconds() or float('inf'))
                    fm_duration = fastest_match.duration_seconds()
                    if fm_duration is not None:
                        ui.label(f"Самая быстрая игра: {fastest_match.p_red} vs {fastest_match.p_blue} ({format_duration(fm_duration)})")
                    
                    # Longest Match
                    longest_match = max(gs.finished_matches, key=lambda m: m.duration_seconds() or float('-inf'))
                    lm_duration = longest_match.duration_seconds()
                    if lm_duration is not None:
                        ui.label(f"Самая долгая игра: {longest_match.p_red} vs {longest_match.p_blue} ({format_duration(lm_duration)})")

                # Best Comeback (simplified: improvement from initial rank to final)
                # Requires gs.initial_ranks to be populated after round 1 (or similar)
                best_comeback_player = None
                max_rank_improvement = -float('inf')
                if gs.initial_ranks:
                    for p_obj in present_players_list:
                        if p_obj.name in gs.initial_ranks:
                            initial_rank = gs.initial_ranks[p_obj.name]
                            final_rank = final_ranks_map.get(p_obj.name, initial_rank) # Use final_ranks_map
                            improvement = initial_rank - final_rank
                            if improvement > max_rank_improvement : # Max positive diff
                                max_rank_improvement = improvement
                                best_comeback_player = p_obj
                    if best_comeback_player and max_rank_improvement > 0:
                         ui.label(f"Лучший камбэк: {best_comeback_player.name} (с {gs.initial_ranks[best_comeback_player.name]} на {final_ranks_map.get(best_comeback_player.name)} место)")
            
            ui.button("Сохранить скриншот", on_click=lambda: ui.screenshot(ui_container)).classes("self-center q-mt-lg")


def build_setup_ui(container: ui.column):
    container.clear() # Clear the main container passed in

    # Define the action for the button first, as it's self-contained logic
    async def start_tournament_action():
        global gs, ui_container # ui_container is the main app container for switching views
        gs.tournament_started = True
        gs.round_idx = 1
        gs.current_round_start_time = datetime.datetime.now()
        
        # Ensure initial ranks are cleared if re-starting setup (though gs is usually fresh or loaded)
        gs.initial_ranks = {} 
        
        gs.match_queue = generate_initial_pairings()
        
        await save_state()
        if ui_container: # Check if the main app container is available
            build_dashboard_ui(ui_container) # Call function to build the next UI
            # Initial seating on all tables after starting
            for table_idx in range(MAX_TABLES):
                if table_idx not in gs.active_matches: # Only if table is free
                    await attempt_to_seat_next_match(table_idx)
            update_scoreboard_display() 
            # Capture initial ranks after the first scoreboard display of round 1
            if gs.round_idx >= 1 and not gs.initial_ranks:
                current_ranks = {p.name: p.current_rank for p in get_present_players() if hasattr(p, 'current_rank')}
                gs.initial_ranks = current_ranks
                await save_state()


    # Build the UI within the provided container
    with container:
        ui.label("Настройка Турнира").classes("text-h4 self-center q-mb-md section-title")
        
        # Player Setup Card
        with ui.card().classes("w-full"):
            ui.label("Участники:").classes("text-h6")
            # Define the label for player count here, so it's parented to this card
            present_count_ui_label = ui.label().classes("q-mb-sm") 
            
            checkbox_grid = ui.grid(columns=3).classes("q-gutter-sm w-full")
            # The start button will be defined after this card.
            # The function update_presence_and_button_state will be defined after the start button.

        # Table Setup Card
        with ui.card().classes("w-full q-mt-md"):
            ui.label("Столы:").classes("text-h6")
            for i in range(MAX_TABLES):
                # Ensure gs.table_names has enough entries
                while len(gs.table_names) <= i:
                    gs.table_names.append(f"Стол {len(gs.table_names) + 1}")
                
                ui.input(f"Название стола {i+1}", value=gs.table_names[i], 
                         on_change=lambda e, idx=i: gs.table_names.__setitem__(idx, e.value))
        
        # Start Button (created as a child of `container` due to `with container:` context)
        the_start_button = ui.button("Начать турнир", on_click=start_tournament_action)
        the_start_button.props("color=primary").classes("q-mt-lg self-center")

        # Now define the function that updates count_label and the_start_button.
        # This function is defined *after* both present_count_ui_label and the_start_button 
        # (which are UI elements) exist in this scope.
        def update_presence_and_button_state():
            count = sum(1 for p_name in PLAYERS_PRESET if gs.players[p_name].present)
            present_count_ui_label.set_text(f"Присутствуют: {count} из {len(PLAYERS_PRESET)}")
            the_start_button.props(f"disable={count < 2}")

        # Populate the checkbox_grid (which is already parented to the first card)
        # and connect their on_change handlers.
        with checkbox_grid: # This re-enters the context of checkbox_grid for adding children
            for name_key in PLAYERS_PRESET: 
                # Ensure player object exists in gs.players; it should from GameState init
                if name_key not in gs.players:
                    gs.players[name_key] = Player(name_key) # Should not be needed if init is robust
                player_obj = gs.players[name_key]
                
                # The lambda calls update_presence_and_button_state, which is now fully defined
                # with its dependencies (present_count_ui_label, the_start_button) in scope.
                ui.checkbox(name_key, value=player_obj.present,
                            on_change=lambda e, p=player_obj: (setattr(p, 'present', e.value), update_presence_and_button_state()))
        
        # Initial call to set the count label and button state correctly based on default/loaded player presence.
        update_presence_and_button_state()


def build_dashboard_ui(container: ui.column):
    container.clear()
    global top_bar_round_label, top_bar_round_time_label, top_bar_tournament_time_label
    global table_cards_display, scoreboard_container

    with container:
        # Top Bar (same as before)
        with ui.row().classes("w-full justify-between items-center q-pa-sm bg-grey-9 text-white q-mb-md"):
            top_bar_round_label = ui.label("Раунд N")
            top_bar_round_time_label = ui.label("⏱ Раунда: 00:00").classes("timer-display")
            top_bar_tournament_time_label = ui.label("До конца: HH:MM:SS").classes("timer-display")

        with ui.splitter(value=60).classes("w-full h-full") as splitter:
            with splitter.before:
                with ui.grid(columns=2 if MAX_TABLES > 1 else 1).classes("w-full q-gutter-md"):
                    for i in range(MAX_TABLES): # i will be table_idx
                        table_name = gs.table_names[i]
                        
                        # Define async handlers for each button, capturing 'i'
                        async def handle_red_click(captured_table_idx=i): # Default argument captures loop var
                            # print(f"DEBUG: Red button clicked for table_idx: {captured_table_idx}")
                            match = gs.active_matches.get(captured_table_idx)
                            if match and match.p_red:
                                await show_confirmation_dialog(captured_table_idx, match.p_red)
                            elif not match:
                                print(f"WARN: Red button clicked, but no active match on table {captured_table_idx}")
                                ui.notify(f"Нет активной игры на столе {gs.table_names[captured_table_idx]}", type='warning')
                            else: # match exists but p_red is None/empty
                                print(f"WARN: Red button clicked, match on table {captured_table_idx} has no p_red: '{match.p_red}'")
                                ui.notify("Ошибка: Имя красного игрока не задано.", type='warning')

                        async def handle_blue_click(captured_table_idx=i): # Default argument captures loop var
                            # print(f"DEBUG: Blue button clicked for table_idx: {captured_table_idx}")
                            match = gs.active_matches.get(captured_table_idx)
                            if match and match.p_blue:
                                await show_confirmation_dialog(captured_table_idx, match.p_blue)
                            elif not match:
                                print(f"WARN: Blue button clicked, but no active match on table {captured_table_idx}")
                                ui.notify(f"Нет активной игры на столе {gs.table_names[captured_table_idx]}", type='warning')
                            else: # match exists but p_blue is None/empty
                                print(f"WARN: Blue button clicked, match on table {captured_table_idx} has no p_blue: '{match.p_blue}'")
                                ui.notify("Ошибка: Имя синего игрока не задано.", type='warning')

                        with ui.card().classes("table-card table-card-idle items-center justify-between") as local_card_element:
                            status_label = ui.label(f"{table_name} - Ожидаем...").classes("text-subtitle1 q-mb-sm")
                            
                            p_red_btn = ui.button("Игрок 1 (Красный)", color="red", on_click=handle_red_click) \
                                .props('disable=true').classes('player-red full-width q-my-xs')
                            
                            p_blue_btn = ui.button("Игрок 2 (Синий)", color="blue", on_click=handle_blue_click) \
                                .props('disable=true').classes('player-blue full-width q-my-xs')
                            
                            timer_label = ui.label("--:--").classes("timer-display self-center q-mt-sm")
                            
                            table_cards_display[i] = {
                                'card': local_card_element,
                                'status_label': status_label,
                                'p_red_btn': p_red_btn,
                                'p_blue_btn': p_blue_btn,
                                'timer_label': timer_label
                            }
                            if i in gs.active_matches:
                                update_table_card_ui_for_match(i, gs.active_matches[i])
                            else:
                                clear_table_card_ui(i) # This calls update_table_card_ui_for_idle
            
            with splitter.after: # Scoreboard part (same as before)
                with ui.column().classes("w-full items-center"):
                    ui.label("Таблица Лидеров").classes("text-h5 section-title q-mb-sm")
                    scoreboard_container = ui.html().classes("w-full")
    
    update_top_bar_timers()
    update_scoreboard_display()

# --- Main App Setup & Page Definition ---
@ui.page('/')
async def main_page(client: Client):
    global ui_container, gs

    # Apply dark theme
    ui.dark_mode().enable()

    # Load state if exists
    # This happens before client connect, so initial gs might be fresh or loaded
    # If loaded state indicates tournament started, build dashboard directly.

    ui_container = ui.column().classes("w-full items-stretch q-pa-md")

    if gs.tournament_finish_screen_shown:
        await show_finish_screen_if_needed() # Rebuilds finish screen
    elif gs.tournament_started:
        build_dashboard_ui(ui_container)
        # Restore active matches UI and attempt to seat on idle tables
        for table_idx in range(MAX_TABLES):
            if table_idx in gs.active_matches:
                 update_table_card_ui_for_match(table_idx, gs.active_matches[table_idx])
            else: # Table is free, try to seat
                 await attempt_to_seat_next_match(table_idx)
        update_scoreboard_display() # Ensure scoreboard is up-to-date
        # If it's the first time displaying scoreboard for round 1 AND initial_ranks not set
        if gs.round_idx >= 1 and not gs.initial_ranks:
            current_ranks = {p.name: p.current_rank for p in get_present_players()}
            gs.initial_ranks = current_ranks # Snapshot ranks for "best comeback"
            await save_state()

    else:
        build_setup_ui(ui_container)

    # Global timers
    ui.timer(1.0, update_top_bar_timers)
    ui.timer(0.5, update_match_timers) # More frequent for match timers
    ui.timer(5.0, lambda: asyncio.create_task(show_finish_screen_if_needed())) # Check for tournament end condition periodically

# --- AutoSave Loop ---
async def auto_save_loop():
    while True:
        await asyncio.sleep(AUTOSAVE_INTERVAL)
        if gs.tournament_started and not gs.tournament_finish_screen_shown :
            await save_state()
            # print(f"Autosaved state at {datetime.datetime.now()}")

# --- App Startup Hook ---
@app.on_startup
async def on_app_startup():
    load_state_sync() # Load state synchronously at startup before first client connects
    asyncio.create_task(auto_save_loop())

# --- App Shutdown Hook ---
@app.on_shutdown
async def on_app_shutdown():
    if gs.tournament_started : # Save one last time on shutdown
        save_state_sync()
        # print("Final state saved on shutdown.")

# --- CSS Styles ---
ui.add_head_html(f'''
<style>
body {{ background:#121212; color:#e0e0e0; font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.q-btn {{ font-size:1.1rem !important; padding:0.6rem 1rem !important; border-radius:8px !important; text-transform: none !important; line-height: 1.2 !important; }}
.q-btn.player-red {{ background:#b71c1c !important; color:#fff !important; }}
.q-btn.player-blue {{ background:#0d47a1 !important; color:#fff !important; }}
.q-card {{ border:1px solid #333; background-color: #1e1e1e; }}
.timer-display {{ font-family: "Consolas", "Monaco", monospace; font-size: 1.2rem; }}
.section-title {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 10px; color: #bbb; }}
.player-card-setup {{ background-color: #2a2a2a; padding: 8px; border-radius: 6px; margin: 4px; text-align: center; }}
.player-card-setup .q-checkbox__label {{ font-size: 1rem; }}
.table-card {{ min-height: 180px; padding: 12px; }} /* Increased min-height */
.table-card-idle {{ background-color: #282828 !important; color: #777 !important; }}
.table-card .q-btn {{ white-space: normal; word-break: break-word; }} /* Allow button text to wrap */
.score-table {{ width: 100%; border-collapse: collapse; }}
.score-table th, .score-table td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }}
.score-table th {{ background-color: #2a2a2a; font-weight: bold; }}
.podium-card {{ padding: 15px; text-align: center; background-color: #222; }}
.podium-1 {{ font-size: 2.8rem; color: gold; }}
.podium-2 {{ font-size: 2.2rem; color: silver; }}
.podium-3 {{ font-size: 1.9rem; color: #cd7f32; }}
.text-grey {{ color: #777 !important; }}
/* Ensure NiceGUI's default checkbox styling is large enough for touch */
.q-checkbox__inner {{ width: 24px !important; height: 24px !important; min-width:24px !important;}}
.q-checkbox__bg {{ width: 100% !important; height: 100% !important;}}
.q-checkbox__label {{ font-size: 1.1rem !important; padding-left: 8px;}}
.q-input .q-field__native {{ font-size: 1.1rem !important; }}
.q-input .q-field__label {{ font-size: 1.1rem !important; }}
</style>
''')

# --- Run the App ---
# For kiosk mode, you might need to launch the browser with specific flags.
# NiceGUI's `native=True` with `fullscreen=True` is the closest for a self-contained app experience.
ui.run(title="Billiard Tournament Console", port=8080, reload=False, native=True, fullscreen=True, window_size=(1280,720))