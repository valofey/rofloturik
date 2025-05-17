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

TOURNAMENT_END_DATETIME = datetime.datetime(2025, 5, 17, 21, 0, 0)
# TOURNAMENT_END_DATETIME = datetime.datetime.now() + datetime.timedelta(minutes=1) # For quick testing

MAX_TABLES = 4

scoreboard_show_details: bool = False
toggle_scoreboard_btn_ref: Optional[ui.button] = None
finish_screen_scoreboard_container: Optional[ui.column] = None


def toggle_scoreboard_details() -> None:
    global scoreboard_show_details, toggle_scoreboard_btn_ref
    scoreboard_show_details = not scoreboard_show_details
    if toggle_scoreboard_btn_ref:
        toggle_scoreboard_btn_ref.props(f"icon={'sym_o_visibility_off' if scoreboard_show_details else 'sym_o_visibility'}")
    update_scoreboard_display()
    if gs.tournament_finish_screen_shown and finish_screen_scoreboard_container:
        original_show_details_temp = scoreboard_show_details
        scoreboard_show_details = True # Force details for final view
        update_scoreboard_display(target_container=finish_screen_scoreboard_container)
        scoreboard_show_details = original_show_details_temp

# --- Data Models ---
class Player:
    def __init__(self, name: str):
        self.name: str = name
        self.present: bool = True
        self.wins: int = 0
        self.losses: int = 0
        self.break_first_count: int = 0
        self.played_with: Set[str] = set()
        self.opponents_defeated: Set[str] = set()
        self.win_durations: List[float] = []
        self.extra_games_played: int = 0
        self.last_played_timestamp: float = 0.0
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
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        self.winner: Optional[str] = None
        self.match_timer_start_monotonic: Optional[float] = None

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
            "match_timer_start_monotonic": self.match_timer_start_monotonic
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Match':
        match = cls(data["p_red"], data["p_blue"], data["table_name"], data["table_idx"])
        match.start_time = datetime.datetime.fromisoformat(data["start_time"]) if data["start_time"] else None
        match.end_time = datetime.datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        match.winner = data["winner"]
        if match.start_time and not match.end_time:
             now_monotonic = time.monotonic()
             time_since_start = (datetime.datetime.now() - match.start_time).total_seconds()
             match.match_timer_start_monotonic = now_monotonic - time_since_start
        return match

class GameState:
    def __init__(self):
        self.players: Dict[str, Player] = {name: Player(name) for name in PLAYERS_PRESET}
        self.table_names: List[str] = [f"Стол {i+1}" for i in range(MAX_TABLES)]
        self.active_matches: Dict[int, Match] = {}
        self.match_queue: List[Tuple[str, Optional[str]]] = []
        self.finished_matches: List[Match] = []
        self.round_idx: int = 0
        self.current_round_start_time: Optional[datetime.datetime] = None
        self.tournament_started: bool = False
        self.tournament_finish_screen_shown: bool = False
        self.played_pairs: Set[frozenset[str]] = set()
        self.last_save_time: float = time.time()
        self.initial_ranks: Dict[str, int] = {}

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
        gs_obj = cls()
        gs_obj.players = {name: Player.from_dict(p_data) for name, p_data in data.get("players", {}).items()}
        for p_name in PLAYERS_PRESET:
            if p_name not in gs_obj.players:
                gs_obj.players[p_name] = Player(p_name)
        gs_obj.table_names = data.get("table_names", [f"Стол {i+1}" for i in range(MAX_TABLES)])
        gs_obj.active_matches = {
            int(idx_str): Match.from_dict(m_data)
            for idx_str, m_data in data.get("active_matches", {}).items()
        }
        gs_obj.match_queue = [tuple(pair) for pair in data.get("match_queue", [])]
        gs_obj.finished_matches = [Match.from_dict(m) for m in data.get("finished_matches", [])]
        gs_obj.round_idx = data.get("round_idx", 0)
        current_round_start_time_str = data.get("current_round_start_time")
        gs_obj.current_round_start_time = datetime.datetime.fromisoformat(current_round_start_time_str) if current_round_start_time_str else None
        gs_obj.tournament_started = data.get("tournament_started", False)
        gs_obj.tournament_finish_screen_shown = data.get("tournament_finish_screen_shown", False)
        gs_obj.played_pairs = set(frozenset(pair_list) for pair_list in data.get("played_pairs", []))
        gs_obj.initial_ranks = data.get("initial_ranks", {})
        return gs_obj

gs = GameState()

# --- UI References ---
ui_root_container: Optional[ui.column] = None
top_bar_round_label: Optional[ui.label] = None
top_bar_round_time_label: Optional[ui.label] = None
top_bar_tournament_time_label: Optional[ui.label] = None
table_cards_display: Dict[int, Dict[str, Any]] = {}
scoreboard_container: Optional[ui.column] = None

# --- State Persistence ---
def save_state_sync():
    global gs
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(gs.to_dict(), f, indent=2, ensure_ascii=False)
        gs.last_save_time = time.time()
    except Exception as e:
        print(f"Error saving state: {e}")

async def save_state():
    save_state_sync()

def load_state_sync():
    global gs
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            gs = GameState.from_dict(data)
            return True
        except Exception as e:
            print(f"Error loading state: {e}. Starting fresh.")
            gs = GameState()
            return False
    else:
        gs = GameState()
        return False


# --- Tournament Logic ---
def get_present_players() -> List[Player]:
    return [p for p_name, p in gs.players.items() if p.present]

def get_present_player_names() -> List[str]:
    return [p.name for p_name, p in gs.players.items() if p.present]

def generate_initial_pairings() -> List[Tuple[str, Optional[str]]]:
    present_names = get_present_player_names()
    if len(present_names) < 2: return []
    best_shuffled_list = list(present_names)
    min_forbidden_count = float('inf')
    for _ in range(200):
        shuffled_list = random.sample(present_names, len(present_names))
        current_forbidden_count = 0
        for i in range(0, len(shuffled_list) - 1 if len(shuffled_list) % 2 == 0 else len(shuffled_list) - 2, 2):
            p1, p2 = shuffled_list[i], shuffled_list[i+1]
            if frozenset({p1, p2}) in FORBIDDEN_PAIRS:
                current_forbidden_count += 1
        if current_forbidden_count < min_forbidden_count:
            min_forbidden_count = current_forbidden_count
            best_shuffled_list = list(shuffled_list)
        if min_forbidden_count == 0: break
    
    pairings = []
    paired_indices = set()
    for i in range(len(best_shuffled_list)):
        if i in paired_indices: continue
        p1_name = best_shuffled_list[i]
        if i + 1 < len(best_shuffled_list):
            p2_name = best_shuffled_list[i+1]
            pairings.append((p1_name, p2_name))
            paired_indices.add(i); paired_indices.add(i+1)
        else: 
            pairings.append((p1_name, None))
            paired_indices.add(i)
            break 
    return pairings


def generate_swiss_pairings() -> List[Tuple[str, Optional[str]]]:
    present_players_obj = get_present_players()
    def sort_key_swiss(p: Player):
        awt = p.average_win_time()
        return (-p.wins, awt if awt is not None else float('inf'), -p.sonnenborn_berger_score(gs.players))
    present_players_obj.sort(key=sort_key_swiss)
    
    sorted_player_names = [p.name for p in present_players_obj]
    new_pairings = []
    available_players = list(sorted_player_names)
    
    while len(available_players) >= 1:
        p1_name = available_players.pop(0)
        found_partner = False
        if not available_players: 
            new_pairings.append((p1_name, None))
            break

        for i in range(len(available_players)):
            p2_name = available_players[i]
            if frozenset({p1_name, p2_name}) not in gs.played_pairs and \
               frozenset({p1_name, p2_name}) not in FORBIDDEN_PAIRS:
                new_pairings.append((p1_name, p2_name))
                available_players.pop(i)
                found_partner = True
                break
        if not found_partner: 
            for i in range(len(available_players)):
                p2_name = available_players[i]
                if frozenset({p1_name, p2_name}) not in gs.played_pairs:
                    new_pairings.append((p1_name, p2_name))
                    available_players.pop(i)
                    found_partner = True
                    print(f"Warning: Swiss pairing ({p1_name}, {p2_name}) is a FORBIDDEN pair but was necessary as they haven't played.")
                    break
        if not found_partner: 
            new_pairings.append((p1_name, None))
    return new_pairings


def choose_floater_for(player_a_name: str) -> Optional[str]:
    eligible_partners = []
    active_player_names = {m.p_red for m in gs.active_matches.values()} | \
                          {m.p_blue for m in gs.active_matches.values()}
    
    for p_name, p_obj in gs.players.items():
        if not p_obj.present or p_name == player_a_name or p_name in active_player_names:
            continue
        eligible_partners.append(p_obj)
    
    if not eligible_partners: return None

    eligible_partners.sort(key=lambda p_sort_obj: (p_sort_obj.extra_games_played, p_sort_obj.last_played_timestamp))
    
    chosen_partner: Optional[Player] = None
    for partner_obj in eligible_partners:
        if frozenset({player_a_name, partner_obj.name}) not in gs.played_pairs and \
           frozenset({player_a_name, partner_obj.name}) not in FORBIDDEN_PAIRS:
            chosen_partner = partner_obj; break
    if not chosen_partner:
        for partner_obj in eligible_partners:
            if frozenset({player_a_name, partner_obj.name}) not in gs.played_pairs:
                chosen_partner = partner_obj
                print(f"Warning: Floater pairing ({player_a_name}, {partner_obj.name}) is FORBIDDEN but they haven't played."); break
    if not chosen_partner:
        for partner_obj in eligible_partners:
            if frozenset({player_a_name, partner_obj.name}) not in FORBIDDEN_PAIRS: 
                chosen_partner = partner_obj
                print(f"Warning: Floater pairing ({player_a_name}, {partner_obj.name}) is a REPEAT (but not forbidden)."); break
    if not chosen_partner and eligible_partners: 
        chosen_partner = eligible_partners[0] 
        print(f"CRITICAL Warning: Floater pairing ({player_a_name}, {chosen_partner.name}) is REPEAT AND FORBIDDEN. Last resort.")
    
    if chosen_partner:
        gs.players[chosen_partner.name].extra_games_played += 1
        gs.players[player_a_name].extra_games_played += 1 
        return chosen_partner.name
    return None

def fair_break(pA_name: str, pB_name: str) -> Tuple[str, str]:
    pA = gs.players[pA_name]
    pB = gs.players[pB_name]
    if pA.break_first_count < pB.break_first_count: red, blue = pA_name, pB_name
    elif pB.break_first_count < pA.break_first_count: red, blue = pB_name, pA_name
    else:
        players_list = [pA_name, pB_name]
        random.shuffle(players_list)
        red, blue = players_list[0], players_list[1]
    gs.players[red].break_first_count += 1
    return red, blue

async def attempt_to_seat_next_match(table_idx: int):
    global gs
    if gs.active_matches.get(table_idx): return

    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    if tournament_time_over and not any(gs.active_matches.values()):
        await show_finish_screen_if_needed(); return
    if tournament_time_over and gs.match_queue:
        update_table_card_ui_for_idle(table_idx, "Время вышло, стол свободен"); return

    if not gs.match_queue:
        if not gs.active_matches:
            new_round_pairings = generate_swiss_pairings()
            if new_round_pairings:
                gs.round_idx += 1
                gs.current_round_start_time = datetime.datetime.now()
                gs.match_queue.extend(new_round_pairings)
                print(f"Advanced to Round {gs.round_idx}. New pairs added: {len(new_round_pairings)}")
                if gs.round_idx == 1 and not gs.initial_ranks:
                    present_players_for_rank = _get_sorted_players_for_scoreboard() 
                    gs.initial_ranks = {player_obj.name: player_obj.current_rank for player_obj in present_players_for_rank}
                await save_state()
                for i in range(MAX_TABLES):
                    if i not in gs.active_matches:
                        asyncio.create_task(attempt_to_seat_next_match(i))
            else:
                print(f"No new pairings generated for round {gs.round_idx + 1}.")
        
        if not gs.match_queue:
            update_table_card_ui_for_idle(table_idx, "Ожидаем следующий раунд"); return

    pair_to_seat_idx = -1
    is_floater_request = False

    for i, (p1, p2_opt) in enumerate(gs.match_queue):
        if p1 and p2_opt:
            current_set = frozenset({p1, p2_opt})
            if current_set not in gs.played_pairs and current_set not in FORBIDDEN_PAIRS:
                pair_to_seat_idx = i; is_floater_request = False; break
    if pair_to_seat_idx == -1:
        for i, (p1, p2_opt) in enumerate(gs.match_queue):
            if p1 and p2_opt is None:
                pair_to_seat_idx = i; is_floater_request = True; break
    if pair_to_seat_idx == -1:
        for i, (p1, p2_opt) in enumerate(gs.match_queue):
            if p1 and p2_opt:
                pair_to_seat_idx = i; is_floater_request = False; break
    
    if pair_to_seat_idx == -1:
        update_table_card_ui_for_idle(table_idx, "Нет доступных пар"); return

    p1_name_raw, p2_name_opt_raw = gs.match_queue.pop(pair_to_seat_idx)
    final_p1_name: str; final_p2_name: Optional[str]

    if is_floater_request:
        final_p1_name = p1_name_raw
        final_p2_name = choose_floater_for(final_p1_name)
        if final_p2_name is None:
            gs.match_queue.append((final_p1_name, None))
            update_table_card_ui_for_idle(table_idx, f"{final_p1_name} ожидает партнера")
            asyncio.create_task(attempt_to_seat_next_match(table_idx))
            return
    else:
        final_p1_name = p1_name_raw
        final_p2_name = p2_name_opt_raw

    if final_p2_name is None:
         print(f"ERROR: Seating {final_p1_name}, P2 is None unexpectedly.")
         if not is_floater_request: gs.match_queue.insert(0, (p1_name_raw, p2_name_opt_raw)) 
         update_table_card_ui_for_idle(table_idx, "Ошибка формирования пары")
         asyncio.create_task(attempt_to_seat_next_match(table_idx)) 
         return

    current_match_players_set = frozenset({final_p1_name, final_p2_name})
    if current_match_players_set in gs.played_pairs and current_match_players_set in FORBIDDEN_PAIRS:
        print(f"CRITICAL SEATING WARNING: Seating REPEAT and FORBIDDEN pair: {final_p1_name} vs {final_p2_name} on table {table_idx}.")
    
    p_red_name, p_blue_name = fair_break(final_p1_name, final_p2_name)
    table_display_name = gs.table_names[table_idx]
    match = Match(p_red=p_red_name, p_blue=p_blue_name, table_name=table_display_name, table_idx=table_idx)
    match.start_time = datetime.datetime.now()
    match.match_timer_start_monotonic = time.monotonic()
    
    gs.active_matches[table_idx] = match
    gs.players[final_p1_name].played_with.add(final_p2_name)
    gs.players[final_p2_name].played_with.add(final_p1_name)
    gs.played_pairs.add(current_match_players_set)

    print(f"Seating match on {table_display_name} (Idx {table_idx}): {p_red_name} (Red) vs {p_blue_name} (Blue)")
    update_table_card_ui_for_match(table_idx, match)
    await save_state()


async def handle_match_result(table_idx: int, winner_name: str):
    global gs
    if table_idx not in gs.active_matches: return

    match = gs.active_matches.pop(table_idx)
    match.end_time = datetime.datetime.now()
    match.winner = winner_name
    gs.finished_matches.append(match)

    winner_obj = gs.players[winner_name]
    loser_name = match.p_red if match.p_blue == winner_name else match.p_blue
    loser_obj = gs.players[loser_name]

    winner_obj.wins += 1
    winner_obj.opponents_defeated.add(loser_name)
    duration_sec = match.duration_seconds()
    if duration_sec is not None: winner_obj.win_durations.append(duration_sec)
    loser_obj.losses += 1
    timestamp_now = time.time()
    winner_obj.last_played_timestamp = timestamp_now
    loser_obj.last_played_timestamp = timestamp_now

    update_scoreboard_display()
    clear_table_card_ui(table_idx)
    await save_state()
    await attempt_to_seat_next_match(table_idx)
    await show_finish_screen_if_needed()

# --- UI Rendering & Updates ---
def format_duration(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"

def update_top_bar_timers():
    if not gs.tournament_started or gs.tournament_finish_screen_shown: return
    if top_bar_round_label:
        top_bar_round_label.set_text(f"Раунд {gs.round_idx if gs.round_idx > 0 else 'Начальный'}")
    if top_bar_round_time_label and gs.current_round_start_time:
        elapsed_seconds = (datetime.datetime.now() - gs.current_round_start_time).total_seconds()
        top_bar_round_time_label.set_text(f"⏱ Раунда: {format_duration(elapsed_seconds)}")
    elif top_bar_round_time_label:
         top_bar_round_time_label.set_text("⏱ Раунда: --:--")
    if top_bar_tournament_time_label:
        now = datetime.datetime.now()
        if now < TOURNAMENT_END_DATETIME:
            remaining_time = TOURNAMENT_END_DATETIME - now
            top_bar_tournament_time_label.set_text(f"До конца: {str(remaining_time).split('.')[0]}")
            top_bar_tournament_time_label.style(remove="color: red;")
        else:
            over_time = now - TOURNAMENT_END_DATETIME
            top_bar_tournament_time_label.set_text(f"Время вышло: -{str(over_time).split('.')[0]}")
            top_bar_tournament_time_label.style(add="color: red;")

def update_match_timers():
    if not gs.tournament_started or gs.tournament_finish_screen_shown: return
    for table_idx, match in gs.active_matches.items():
        if table_idx in table_cards_display and match.match_timer_start_monotonic is not None:
            elapsed = match.elapsed_seconds()
            table_cards_display[table_idx]['timer_label'].set_text(format_duration(elapsed))

def update_table_card_ui_for_match(table_idx: int, match: Match):
    card_elements = table_cards_display.get(table_idx)
    if not card_elements: return
    p_red_btn_el = card_elements.get('p_red_btn'); p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label'); timer_label_el = card_elements.get('timer_label')
    card_el = card_elements.get('card')
    if not all([p_red_btn_el, p_blue_btn_el, status_label_el, timer_label_el, card_el]): return

    p_red_btn_el.set_text(match.p_red) 
    p_red_btn_el.props('disable=false flat=false unelevated=false').classes(remove='button-idle', add='player-red') 

    p_blue_btn_el.set_text(match.p_blue) 
    p_blue_btn_el.props('disable=false flat=false unelevated=false').classes(remove='button-idle', add='player-blue') 

    status_label_el.set_text(match.table_name) 
    status_label_el.classes(remove='text-grey') 

    timer_label_el.set_text("00:00") 
    card_el.classes(remove='table-card-idle') 

def clear_table_card_ui(table_idx: int):
    update_table_card_ui_for_idle(table_idx, "Ожидаем пару...")

def update_table_card_ui_for_idle(table_idx: int, message: str):
    card_elements = table_cards_display.get(table_idx)
    if not card_elements: return
    p_red_btn_el = card_elements.get('p_red_btn'); p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label'); timer_label_el = card_elements.get('timer_label')
    card_el = card_elements.get('card')
    if not all([p_red_btn_el, p_blue_btn_el, status_label_el, timer_label_el, card_el]): return

    p_red_btn_el.set_text("---") 
    p_red_btn_el.props('flat unelevated disable=true').classes('button-idle', remove='player-red') 

    p_blue_btn_el.set_text("---") 
    p_blue_btn_el.props('flat unelevated disable=true').classes('button-idle', remove='player-blue') 

    status_label_el.set_text(message) 
    status_label_el.classes(add='text-grey') 

    timer_label_el.set_text("--:--") 
    card_el.classes(add='table-card-idle') 

def _get_sorted_players_for_scoreboard() -> List[Player]:
    present_players_list = get_present_players()
    def sort_key_scoreboard(p_sort_obj: Player):
        awt = p_sort_obj.average_win_time()
        sb_score = p_sort_obj.sonnenborn_berger_score(gs.players)
        return (-p_sort_obj.wins, awt if awt is not None else float('inf'), -sb_score)
    present_players_list.sort(key=sort_key_scoreboard)
    ranks = {}
    for i, p_obj in enumerate(present_players_list):
        key_curr = sort_key_scoreboard(p_obj)
        if i == 0: ranks[p_obj.name] = 1
        else:
            key_prev = sort_key_scoreboard(present_players_list[i-1])
            ranks[p_obj.name] = ranks[present_players_list[i-1].name] if key_curr == key_prev else i + 1
        p_obj.current_rank = ranks[p_obj.name]
    return present_players_list

def update_scoreboard_display(target_container: Optional[ui.column] = None):
    global scoreboard_show_details
    current_scoreboard_container = target_container if target_container else scoreboard_container

    if not current_scoreboard_container:
        return

    if not gs.tournament_started and not gs.tournament_finish_screen_shown :
        current_scoreboard_container.clear()
        with current_scoreboard_container:
            ui.label("Турнир не начат").classes("text-center full-width q-pa-md text-grey")
        return

    present_players_list = _get_sorted_players_for_scoreboard()
    current_scoreboard_container.clear()

    with current_scoreboard_container:
        if not present_players_list:
            ui.label("Нет игроков для отображения").classes("text-center full-width q-pa-md text-grey")
            return

        with ui.row().classes("w-full scoreboard-header-row items-center no-wrap q-px-sm"):
            ui.label("#").style("width: 10%; text-align: left; min-width: 20px;")
            name_header_width = "35%" if scoreboard_show_details else "75%"
            ui.label("Имя").style(f"width: {name_header_width}; text-align: left; min-width: 70px;")
            ui.label("П").style("width: 10%; text-align: center; min-width: 20px;").tooltip("Побед")
            if scoreboard_show_details:
                ui.label("⏱️").style("width: 20%; text-align: center; min-width: 40px;").tooltip("Среднее время победы")
                ui.label("СБ").style("width: 15%; text-align: center; min-width: 30px;").tooltip("Коэф. Зоннеборна-Бергера")

        for p_obj in present_players_list:
            with ui.card().classes("w-full scoreboard-player-card no-shadow flat bordered"):
                with ui.row().classes("w-full items-center no-wrap"): 
                    ui.label(f"{p_obj.current_rank}").style("width: 10%; text-align: left; font-weight: bold; min-width: 20px;")
                    
                    name_col_width = "35%" if scoreboard_show_details else "75%"
                    ui.label(p_obj.name).style(f"width: {name_col_width}; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 70px;").tooltip(p_obj.name)
                    
                    ui.label(f"{p_obj.wins}").style("width: 10%; text-align: center; font-weight: bold; min-width: 20px;")
                    
                    if scoreboard_show_details:
                        avg_win_time_str = format_duration(p_obj.average_win_time()) if p_obj.average_win_time() is not None else "-"
                        sb_str = f"{p_obj.sonnenborn_berger_score(gs.players):.1f}"
                        ui.label(avg_win_time_str).style("width: 20%; text-align: center; min-width: 40px;")
                        ui.label(sb_str).style("width: 15%; text-align: center; min-width: 30px;")

async def show_confirmation_dialog(table_idx: int, player_name: str):
    try:
        table_display_name = gs.table_names[table_idx] if 0 <= table_idx < len(gs.table_names) else f"Стол {table_idx + 1}"
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Победил {player_name} на столе {table_display_name}?")
            with ui.row().classes("justify-end w-full"):
                ui.button("Нет", on_click=lambda: dialog.submit(False), color='negative')
                ui.button("Да", on_click=lambda: dialog.submit(True), color='positive')
        result = await dialog
        if result:
            await handle_match_result(table_idx, player_name)
    except Exception as e:
        print(f"ERROR in show_confirmation_dialog: {e}")
        ui.notify(f"Ошибка диалога: {e}", type='negative', multi_line=True, close_button=True)

async def show_finish_screen_if_needed():
    global gs, ui_root_container, finish_screen_scoreboard_container, scoreboard_show_details
    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    no_active_matches = not gs.active_matches

    if tournament_time_over and no_active_matches and gs.tournament_started and not gs.tournament_finish_screen_shown:
        gs.tournament_finish_screen_shown = True
        await save_state()
        if ui_root_container is None: return

        ui_root_container.clear()
        with ui_root_container:
            with ui.splitter(value=50).classes("w-full finish-splitter") as finish_splitter:
                with finish_splitter.before:
                    with ui.column().classes("w-full items-center q-pa-md overflow-auto flex-1 min-h-0"):
                        ui.label("Турнир Завершен!").classes("text-h3 self-center q-my-md")
                        present_players_list = _get_sorted_players_for_scoreboard()
                        final_ranks_map = {p_obj.name: p_obj.current_rank for p_obj in present_players_list}
                        with ui.row().classes("justify-center q-gutter-md q-my-lg"):
                            if len(present_players_list) >= 1:
                                p1 = present_players_list[0]
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥇").classes("podium-1"); ui.label(f"{p1.name}").classes("text-h5"); ui.label(f"{p1.wins} побед").classes("text-caption")
                            p2 = next((p_obj for p_obj in present_players_list if final_ranks_map.get(p_obj.name) == 2), None) if len(present_players_list) >=2 else None
                            if not p2 and len(present_players_list) > 1 and final_ranks_map.get(present_players_list[1].name) != 1: p2 = present_players_list[1]
                            if p2:
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥈").classes("podium-2"); ui.label(f"{p2.name}").classes("text-h6"); ui.label(f"{p2.wins} побед").classes("text-caption")
                            p3 = next((p_obj for p_obj in present_players_list if final_ranks_map.get(p_obj.name) == 3), None) if len(present_players_list) >=3 else None
                            if not p3 and len(present_players_list) >=3:
                                distinct_ranks_seen, last_rank = 0,0
                                for p_cand in present_players_list:
                                    curr_rank = final_ranks_map.get(p_cand.name,0)
                                    if curr_rank != last_rank: distinct_ranks_seen+=1; last_rank = curr_rank
                                    if distinct_ranks_seen == 3: p3 = p_cand; break
                            if p3:        
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥉").classes("podium-3"); ui.label(f"{p3.name}").classes("text-subtitle1"); ui.label(f"{p3.wins} побед").classes("text-caption")
                        
                        with ui.card().classes("q-my-md w-full"):
                            ui.label("Статистика турнира:").classes("text-h6 q-mb-sm")
                            if gs.finished_matches:
                                ui.label(f"Всего сыграно матчей: {len(gs.finished_matches)}")
                                valid_durations = [m.duration_seconds() for m in gs.finished_matches if m.duration_seconds() is not None and m.duration_seconds() > 0]
                                if valid_durations: ui.label(f"Средняя длительность матча: {format_duration(sum(valid_durations) / len(valid_durations))}")
                                if gs.finished_matches : 
                                    fastest_match = min(gs.finished_matches, key=lambda m: m.duration_seconds() or float('inf'))
                                    fm_duration = fastest_match.duration_seconds()
                                    if fm_duration is not None: ui.label(f"Самый быстрый матч: {fastest_match.p_red} vs {fastest_match.p_blue} ({format_duration(fm_duration)})")
                                    longest_match = max(gs.finished_matches, key=lambda m: m.duration_seconds() or float('-inf'))
                                    lm_duration = longest_match.duration_seconds()
                                    if lm_duration is not None: ui.label(f"Самый долгий матч: {longest_match.p_red} vs {longest_match.p_blue} ({format_duration(lm_duration)})")
                            most_games_count = -1; most_games_player_name = "-"
                            for p_obj_stat in get_present_players():
                                games_played = p_obj_stat.wins + p_obj_stat.losses
                                if games_played > most_games_count: most_games_count, most_games_player_name = games_played, p_obj_stat.name
                                elif games_played == most_games_count: most_games_player_name += f", {p_obj_stat.name}"
                            if most_games_count > 0: ui.label(f"Больше всего матчей сыграл(и): {most_games_player_name} ({most_games_count})")
                            extra_gamers = [p_obj.name for p_obj in gs.players.values() if p_obj.present and p_obj.extra_games_played > 0] 
                            if extra_gamers: ui.label(f"Сыграли доп. игры (флоутеры): {', '.join(extra_gamers)}")
                            best_comeback_player, max_rank_improvement = None, -float('inf')
                            if gs.initial_ranks:
                                for p_obj_cb in present_players_list:
                                    if p_obj_cb.name in gs.initial_ranks:
                                        initial_r, final_r = gs.initial_ranks[p_obj_cb.name], final_ranks_map.get(p_obj_cb.name, gs.initial_ranks[p_obj_cb.name])
                                        improvement = initial_r - final_r
                                        if improvement > max_rank_improvement: max_rank_improvement, best_comeback_player = improvement, p_obj_cb
                                if best_comeback_player and max_rank_improvement > 0:
                                     ui.label(f"Лучший камбэк: {best_comeback_player.name} (с {gs.initial_ranks[best_comeback_player.name]} на {final_ranks_map.get(best_comeback_player.name)} место, улучшение на {max_rank_improvement} поз.)")
                
                with finish_splitter.after:
                    with ui.column().classes("w-full items-center q-pa-md overflow-auto flex-1 min-h-0"):
                        ui.label("Итоговая Таблица Лидеров").classes("text-h5 section-title q-mb-sm")
                        finish_screen_scoreboard_container = ui.column().classes("w-full scoreboard-card-list-container")
                        original_show_details_temp = scoreboard_show_details
                        scoreboard_show_details = True # Always show details on finish screen
                        update_scoreboard_display(target_container=finish_screen_scoreboard_container)
                        scoreboard_show_details = original_show_details_temp # Restore for dashboard if ever revisited
                        
def build_setup_ui(container: ui.column):
    container.clear()
    async def start_tournament_action():
        global gs, ui_root_container
        gs.tournament_started = True
        gs.round_idx = 0 
        gs.current_round_start_time = datetime.datetime.now()
        gs.initial_ranks = {} 
        gs.match_queue = generate_initial_pairings()
        
        present_players_for_rank = _get_sorted_players_for_scoreboard() 
        gs.initial_ranks = {player_obj.name: player_obj.current_rank for player_obj in present_players_for_rank}
        
        await save_state()
        if ui_root_container:
            build_dashboard_ui(ui_root_container)
            for table_idx in range(MAX_TABLES):
                if table_idx not in gs.active_matches:
                    await attempt_to_seat_next_match(table_idx)
            update_scoreboard_display() 

    with container:
        ui.label("Настройка Турнира").classes("text-h4 self-center q-mb-md section-title")
        with ui.card().classes("w-full"):
            ui.label("Участники:").classes("text-h6")
            present_count_ui_label = ui.label().classes("q-mb-sm")
            checkbox_grid = ui.grid(columns=3).classes("q-gutter-sm w-full")
        with ui.card().classes("w-full q-mt-md"):
            ui.label("Столы:").classes("text-h6")
            for i in range(MAX_TABLES):
                while len(gs.table_names) <= i: gs.table_names.append(f"Стол {len(gs.table_names) + 1}")
                ui.input(f"Название стола {i+1}", value=gs.table_names[i],
                         on_change=lambda e, idx=i: gs.table_names.__setitem__(idx, e.value))
        the_start_button = ui.button("Начать турнир", on_click=start_tournament_action).props("color=primary").classes("q-mt-lg self-center")

        def update_presence_and_button_state():
            count = sum(1 for p_name, p_obj in gs.players.items() if p_obj.present)
            present_count_ui_label.set_text(f"Присутствуют: {count} из {len(PLAYERS_PRESET)}")
            the_start_button.props(f"disable={count < 2}")

        with checkbox_grid:
            for name_key in PLAYERS_PRESET:
                if name_key not in gs.players: gs.players[name_key] = Player(name_key) 
                player_obj_cb = gs.players[name_key]
                ui.checkbox(name_key, value=player_obj_cb.present,
                            on_change=lambda e, p_cb=player_obj_cb: (setattr(p_cb, 'present', e.value), update_presence_and_button_state()))
        update_presence_and_button_state()


def build_dashboard_ui(container: ui.column):
    container.clear()
    global top_bar_round_label, top_bar_round_time_label, top_bar_tournament_time_label
    global table_cards_display, scoreboard_container, toggle_scoreboard_btn_ref

    with container:
        with ui.row().classes("w-full justify-between items-center q-pa-sm bg-grey-9 text-white q-mb-md"): 
            top_bar_round_label = ui.label("Раунд N")
            top_bar_round_time_label = ui.label("⏱ Раунда: 00:00").classes("timer-display")
            top_bar_tournament_time_label = ui.label("До конца: HH:MM:SS").classes("timer-display")

        with ui.splitter(value=60).classes("w-full dashboard-splitter") as splitter: 
            with splitter.before:
                with ui.column().classes("w-full q-pa-md flex-1 min-h-0 overflow-auto"): 
                    with ui.grid(columns=2 if MAX_TABLES > 1 else 1).classes("w-full q-gutter-md"):
                        for i in range(MAX_TABLES):
                            table_name = gs.table_names[i]
                            async def handle_red_click(captured_table_idx=i):
                                match = gs.active_matches.get(captured_table_idx)
                                if match and match.p_red: await show_confirmation_dialog(captured_table_idx, match.p_red)
                            async def handle_blue_click(captured_table_idx=i):
                                match = gs.active_matches.get(captured_table_idx)
                                if match and match.p_blue: await show_confirmation_dialog(captured_table_idx, match.p_blue)
                            with ui.card().classes("table-card table-card-idle items-center justify-between") as local_card_element:
                                status_label = ui.label(f"{table_name} - Ожидаем...").classes("text-subtitle1 q-mb-sm text-center")
                                p_red_btn = ui.button("---", color="red", on_click=handle_red_click).props('flat unelevated disable=true').classes('button-idle full-width q-my-xs')
                                p_blue_btn = ui.button("---", color="blue", on_click=handle_blue_click).props('flat unelevated disable=true').classes('button-idle full-width q-my-xs')
                                timer_label = ui.label("--:--").classes("timer-display self-center q-mt-sm")
                                table_cards_display[i] = {'card': local_card_element, 'status_label': status_label, 'p_red_btn': p_red_btn, 'p_blue_btn': p_blue_btn, 'timer_label': timer_label}
                                if i in gs.active_matches: update_table_card_ui_for_match(i, gs.active_matches[i])
                                else: clear_table_card_ui(i)
            
            with splitter.after:
                with ui.column().classes("w-full items-center q-pa-xs flex-1 min-h-0 overflow-auto"):
                    with ui.row().classes("w-full justify-between items-center"):
                         ui.label("Таблица Лидеров").classes("text-h5 section-title q-mb-sm")
                         toggle_scoreboard_btn_ref = ui.button(
                             icon='sym_o_visibility_off' if scoreboard_show_details else 'sym_o_visibility', 
                             on_click=toggle_scoreboard_details
                         ).props("flat dense round").tooltip("Показать/Скрыть детали")
                    scoreboard_container = ui.column().classes("w-full scoreboard-card-list-container") 
    update_top_bar_timers()
    update_scoreboard_display()

# --- Main App Setup & Page Definition ---
@ui.page('/')
async def main_page(client: Client):
    global ui_root_container, gs
    ui.dark_mode().enable()
    
    ui_root_container = ui.column().classes("w-full flex-1 min-h-0 items-stretch q-pa-md main-container overflow-hidden")

    if gs.tournament_finish_screen_shown:
        await show_finish_screen_if_needed()
    elif gs.tournament_started:
        build_dashboard_ui(ui_root_container)
        for table_idx in range(MAX_TABLES):
            if table_idx in gs.active_matches: 
                update_table_card_ui_for_match(table_idx, gs.active_matches[table_idx])
            else: 
                await attempt_to_seat_next_match(table_idx)
        update_scoreboard_display() 
        if gs.round_idx == 0 and not gs.initial_ranks:
            present_players_for_rank = _get_sorted_players_for_scoreboard()
            gs.initial_ranks = {player_obj.name: player_obj.current_rank for player_obj in present_players_for_rank}
            await save_state()
    else:
        build_setup_ui(ui_root_container)

    ui.timer(1.0, update_top_bar_timers)
    ui.timer(0.5, update_match_timers)
    ui.timer(5.0, lambda: asyncio.create_task(show_finish_screen_if_needed()))

# --- AutoSave Loop ---
async def auto_save_loop():
    while True:
        await asyncio.sleep(AUTOSAVE_INTERVAL)
        if gs.tournament_started and not gs.tournament_finish_screen_shown:
            await save_state()

# --- App Startup Hook ---
@app.on_startup
async def on_app_startup():
    load_state_sync() 
    if toggle_scoreboard_btn_ref and gs:
         toggle_scoreboard_btn_ref.props(f"icon={'sym_o_visibility_off' if scoreboard_show_details else 'sym_o_visibility'}")
    asyncio.create_task(auto_save_loop())

# --- App Shutdown Hook ---
@app.on_shutdown
async def on_app_shutdown():
    if gs and gs.tournament_started:
        save_state_sync()

# --- Consolidated CSS Styles ---
consolidated_css = f'''
<style>
/* --- Global Resets & Base --- */
html {{
    margin: 0;
    padding: 0;
    overflow: hidden !important; /* CRITICAL: Prevent html scrollbar */
    height: 100vh;
    width: 100vw;
    box-sizing: border-box;
}}
body {{
    background: #121212;
    color: #e0e0e0;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    overflow: hidden !important; /* CRITICAL: Prevent body scrollbar */
    height: 100%; /* Fill 100% of html (which is 100vh) */
    width: 100%;
    display: flex; /* Make body a flex container */
    flex-direction: column; /* Stack children vertically */
    box-sizing: border-box;
}}
*, *::before, *::after {{
    box-sizing: border-box;
}}

/* Ensure Quasar's typical app structure also propagates flex sizing */
#q-app, .q-layout, .q-page-container, .q-page {{
    flex: 1 1 auto; 
    display: flex;
    flex-direction: column;
    min-height: 0; 
    width: 100%; 
}}
.q-page {{
    overflow: visible; 
}}


/* --- Main Application Container --- */
.main-container {{ /* This is your ui_root_container */
    flex: 1 1 auto; 
    min-height: 0;  
    /* width: 100%; /* From w-full class */
    /* display: flex; /* From ui.column */
    /* flex-direction: column; /* From ui.column */
    /* q-pa-md adds padding. */
    /* overflow-hidden is applied via .classes() in Python */
}}

/* --- Splitter Styling --- */
.dashboard-splitter,
.finish-splitter {{
    flex-grow: 1;
    flex-shrink: 1;
    flex-basis: 0; 
    min-height: 0; 
    overflow: hidden !important; 
    width: 100%;
}}

/* --- Splitter Panels (.q-splitter__panel, which are q-splitter__before and q-splitter__after) --- */
.q-splitter__panel {{
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    overflow: hidden !important; 
}}

/* --- Content within Splitter Panels (Your ui.column elements) --- */
.q-splitter__panel > .q-column {{ 
    flex: 1 1 auto; 
    min-height: 0;  
    /* overflow: auto; /* Applied via .classes('overflow-auto') in Python where needed */
}}


/* --- General Utility --- */
.overflow-auto {{
    overflow: auto;
}}

/* --- Existing Component Styles --- */
.q-btn {{
    font-size:1.0rem !important;
    padding:0.5rem 0.8rem !important;
    border-radius:6px !important;
    text-transform: none !important;
    line-height: 1.2 !important;
    min-height: 40px !important;
    white-space: normal; word-break: break-word;
}}
.q-btn.player-red {{ background:#b71c1c !important; color:#fff !important; }}
.q-btn.player-blue {{ background:#0d47a1 !important; color:#fff !important; }}
.q-btn.button-idle {{
    background-color: #333 !important; color: #666 !important;
    box-shadow: none !important; border: 1px solid #444 !important;
}}

.q-card {{ border:1px solid #333; background-color: #1e1e1e; }}
.timer-display {{ font-family: "Consolas", "Monaco", monospace; font-size: 1.2rem; }}
.section-title {{ font-size: 1.5rem; font-weight: bold; margin-bottom: 10px; color: #bbb; }}

.table-card {{
    min-height: 190px;
    padding: 10px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: space-between;
}}
.table-card-idle {{ background-color: #282828 !important; color: #777 !important; }}
.table-card .q-btn {{ width: 90%; }}

.scoreboard-card-list-container.q-column {{
    gap: 0px; /* Adjusted from 1px to 0px */
}}
.scoreboard-player-card.q-card {{
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #444;
    background-color: #262626;
}}
.scoreboard-player-card .q-row {{
    min-height: 24px;
    align-items: center;
}}
.scoreboard-player-card .q-label {{
    font-size: 0.8rem;
    line-height: 1.0;
}}
.scoreboard-header-row.q-row {{
    min-height: 24px;
}}
.scoreboard-header-row .q-label {{
    font-size: 0.7rem;
    font-weight: bold;
    color: #9e9e9e;
    text-transform: uppercase;
}}

.podium-card {{ padding: 15px; text-align: center; background-color: #222; min-width: 120px; }}
.podium-1 {{ font-size: 2.8rem; color: gold; }}
.podium-2 {{ font-size: 2.2rem; color: silver; }}
.podium-3 {{ font-size: 1.9rem; color: #cd7f32; }}

.text-grey {{ color: #777 !important; }}
.q-checkbox__inner {{ width: 22px !important; height: 22px !important; min-width:22px !important;}}
.q-checkbox__bg {{ width: 100% !important; height: 100% !important;}}
.q-checkbox__label {{ font-size: 1.0rem !important; padding-left: 6px;}}
.q-input .q-field__native {{ font-size: 1.0rem !important; }}
.q-input .q-field__label {{ font-size: 1.0rem !important; }}
</style>
'''
ui.add_head_html(consolidated_css)


ui.run(title="Billiard Tournament Console", port=8080, reload=False, native=True, fullscreen=True, window_size=(1280,720))