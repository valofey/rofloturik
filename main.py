import asyncio
import datetime
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import math

from nicegui import app, ui, Client

#--- Constants ---

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
# TOURNAMENT_END_DATETIME = datetime.datetime(2025, 5, 17, 21, 0, 0)
# For testing, you might want to adjust this year/date or make it more dynamic.
TOURNAMENT_END_DATETIME = datetime.datetime.now() + datetime.timedelta(minutes=2) # For quick testing (2 min)

MAX_TABLES = 4
scoreboard_show_details: bool = False      # по-умолчанию 3-колоночный вид (Рейтинг, Имя, Побед)
toggle_scoreboard_btn_ref: Optional[ui.button] = None # Ref to the toggle button for updating its icon

def toggle_scoreboard_details() -> None:
    """Колбэк для кнопки ‟Показать/Скрыть детали”."""
    global scoreboard_show_details, toggle_scoreboard_btn_ref
    scoreboard_show_details = not scoreboard_show_details
    if toggle_scoreboard_btn_ref:
        toggle_scoreboard_btn_ref.props(f"icon={'sym_o_visibility_off' if scoreboard_show_details else 'sym_o_visibility'}")
    update_scoreboard_display() # перерисовать таблицу
    if gs.tournament_finish_screen_shown and finish_screen_scoreboard_container:
        # If finish screen is active, update its scoreboard too
        # (though it's usually forced to detailed view)
        original_show_details_temp = scoreboard_show_details
        scoreboard_show_details = True # Keep finish screen detailed
        update_scoreboard_display(target_container=finish_screen_scoreboard_container)
        scoreboard_show_details = original_show_details_temp


#--- Data Models ---

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
        self.active_matches: Dict[int, Match] = {} # table_idx -> Match
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
        gs = cls()
        gs.players = {name: Player.from_dict(p_data) for name, p_data in data.get("players", {}).items()}
        for p_name in PLAYERS_PRESET:
            if p_name not in gs.players:
                gs.players[p_name] = Player(p_name)

        gs.table_names = data.get("table_names", [f"Стол {i+1}" for i in range(MAX_TABLES)])
        gs.active_matches = {
            int(idx_str): Match.from_dict(m_data)
            for idx_str, m_data in data.get("active_matches", {}).items()
        }
        gs.match_queue = [tuple(pair) for pair in data.get("match_queue", [])]
        gs.finished_matches = [Match.from_dict(m_data) for m_data in data.get("finished_matches", [])]
        gs.round_idx = data.get("round_idx", 0)
        current_round_start_time_str = data.get("current_round_start_time")
        gs.current_round_start_time = datetime.datetime.fromisoformat(current_round_start_time_str) if current_round_start_time_str else None
        gs.tournament_started = data.get("tournament_started", False)
        gs.tournament_finish_screen_shown = data.get("tournament_finish_screen_shown", False)
        gs.played_pairs = set(frozenset(pair_list) for pair_list in data.get("played_pairs", []))
        gs.initial_ranks = data.get("initial_ranks", {})
        return gs

gs = GameState()

#--- UI References ---
ui_container: Optional[ui.column] = None
top_bar_round_label: Optional[ui.label] = None
top_bar_round_time_label: Optional[ui.label] = None
top_bar_tournament_time_label: Optional[ui.label] = None
table_cards_display: Dict[int, Dict[str, Any]] = {}
scoreboard_container: Optional[ui.html] = None
finish_screen_scoreboard_container: Optional[ui.html] = None # For the separate scoreboard on finish screen

#--- State Persistence ---
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
    return False

#--- Tournament Logic ---
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

    for _ in range(200): # Try multiple shuffles to minimize forbidden pairs
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
    paired_indices = set()
    for i in range(num_to_pair):
        if i in paired_indices:
            continue
        p1_name = best_shuffled_list[i]

        if i + 1 < num_to_pair:
            p2_name = best_shuffled_list[i+1]
            pairings.append((p1_name, p2_name))
            paired_indices.add(i)
            paired_indices.add(i+1)
        else:
            pairings.append((p1_name, None)) # Bye for the last player if odd number
            paired_indices.add(i)
            break
    return pairings

def generate_swiss_pairings() -> List[Tuple[str, Optional[str]]]:
    present_players_obj = get_present_players()

    def sort_key_swiss(p: Player):
        awt = p.average_win_time()
        return (
            -p.wins,
            awt if awt is not None else float('inf'), # Lower (faster) avg win time is better
            -p.sonnenborn_berger_score(gs.players)
        )

    present_players_obj.sort(key=sort_key_swiss)
    sorted_player_names = [p.name for p in present_players_obj]
    new_pairings = []
    available_players = list(sorted_player_names)

    while len(available_players) >= 1:
        p1_name = available_players.pop(0)
        found_partner = False
        if not available_players: # p1 is the last one, gets a bye or waits for floater
            new_pairings.append((p1_name, None))
            break

        # Try to find an opponent they haven't played and is not forbidden
        for i in range(len(available_players)):
            p2_name = available_players[i]
            if frozenset({p1_name, p2_name}) not in gs.played_pairs and \
               frozenset({p1_name, p2_name}) not in FORBIDDEN_PAIRS:
                new_pairings.append((p1_name, p2_name))
                available_players.pop(i)
                found_partner = True
                break

        if not found_partner: # If no such opponent, try one they haven't played (even if forbidden)
            for i in range(len(available_players)):
                p2_name = available_players[i]
                if frozenset({p1_name, p2_name}) not in gs.played_pairs:
                    new_pairings.append((p1_name, p2_name))
                    print(f"Warning: Swiss pairing ({p1_name}, {p2_name}) is a FORBIDDEN pair but necessary as no other non-played partner found.")
                    available_players.pop(i)
                    found_partner = True
                    break

        if not found_partner: # If all available have been played, this means a repeat is necessary.
                              # Or p1_name is left alone. For Swiss, usually p1_name gets a bye.
            new_pairings.append((p1_name, None)) # This player will wait or get a bye.
                                                # Floater logic will handle (Player, None) from queue.
    return new_pairings


def choose_floater_for(player_a_name: str) -> Optional[str]:
    eligible_partners = []
    # player_a_obj = gs.players[player_a_name] # Not strictly needed here
    active_player_names = {m.p_red for m in gs.active_matches.values()} | \
                          {m.p_blue for m in gs.active_matches.values()}

    for p_name, p_obj in gs.players.items():
        if not p_obj.present or p_name == player_a_name or p_name in active_player_names:
            continue
        # Check if p_name is already in the match_queue as P1 of a (P1,P2) pair or P1 of (P1,None)
        # to avoid pulling someone who is already slated for a specific match or bye.
        is_in_queue_as_p1 = any(item[0] == p_name for item in gs.match_queue)
        if is_in_queue_as_p1:
            continue

        eligible_partners.append(p_obj)

    if not eligible_partners:
        return None

    # Sort by fewest extra games, then longest wait time (last_played_timestamp ASC)
    eligible_partners.sort(key=lambda p: (p.extra_games_played, p.last_played_timestamp))

    chosen_partner = None
    # Priority 1: Not played with A AND not forbidden
    for partner_obj in eligible_partners:
        if frozenset({player_a_name, partner_obj.name}) not in gs.played_pairs and \
           frozenset({player_a_name, partner_obj.name}) not in FORBIDDEN_PAIRS:
            chosen_partner = partner_obj
            break

    # Priority 2: Not played with A (even if forbidden)
    if not chosen_partner:
        for partner_obj in eligible_partners:
            if frozenset({player_a_name, partner_obj.name}) not in gs.played_pairs:
                chosen_partner = partner_obj
                print(f"Warning: Floater pairing ({player_a_name}, {partner_obj.name}) is a FORBIDDEN pair but chosen as no other non-played available.")
                break

    # Priority 3: Played, but not forbidden (least recent repeat, fewest extra games)
    if not chosen_partner:
        for partner_obj in eligible_partners: # Already sorted by extra_games_played, last_played_timestamp
            if frozenset({player_a_name, partner_obj.name}) not in FORBIDDEN_PAIRS:
                chosen_partner = partner_obj
                print(f"Warning: Floater pairing ({player_a_name}, {partner_obj.name}) is a REPEAT pairing (but not forbidden).")
                break

    # Priority 4: Played AND forbidden (least recent repeat, fewest extra games) - last resort
    if not chosen_partner and eligible_partners: # Take the top sorted one if no other choice
        chosen_partner = eligible_partners[0]
        print(f"CRITICAL Warning: Floater pairing ({player_a_name}, {chosen_partner.name}) is a REPEAT AND FORBIDDEN pair. Last resort.")


    if chosen_partner:
        gs.players[chosen_partner.name].extra_games_played += 1
        gs.players[player_a_name].extra_games_played += 1 # Player A also plays an "extra" game (as floater host)
        return chosen_partner.name
    return None

def fair_break(pA_name: str, pB_name: str) -> Tuple[str, str]:
    pA = gs.players[pA_name]
    pB = gs.players[pB_name]

    if pA.break_first_count < pB.break_first_count:
        red, blue = pA_name, pB_name
    elif pB.break_first_count < pA.break_first_count:
        red, blue = pB_name, pA_name
    else: # Equal break counts, randomize
        players_list = [pA_name, pB_name]
        random.shuffle(players_list)
        red, blue = players_list[0], players_list[1]

    gs.players[red].break_first_count += 1
    return red, blue

async def attempt_to_seat_next_match(table_idx: int):
    global gs
    if gs.active_matches.get(table_idx): # Table is busy
        return

    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    if tournament_time_over and not any(gs.active_matches.values()): # Tournament over, all matches finished
        await show_finish_screen_if_needed()
        return
    if tournament_time_over and gs.match_queue: # Time is over, don't seat new matches from queue
        update_table_card_ui_for_idle(table_idx, "Время вышло, стол свободен")
        # Potentially clear queue or handle players waiting for byes if that's a desired feature
        return

    if not gs.match_queue: # Queue is empty
        if not gs.active_matches: # And no tables are active, try to generate new round
            new_round_pairings = generate_swiss_pairings()
            if new_round_pairings:
                gs.match_queue.extend(new_round_pairings)
                gs.round_idx += 1
                gs.current_round_start_time = datetime.datetime.now()
                # Capture initial ranks after first real round pairings are made
                if gs.round_idx == 1 and not gs.initial_ranks:
                    update_scoreboard_display() # Calculate ranks based on initial state
                    current_ranks = {p.name: p.current_rank for p in get_present_players()}
                    gs.initial_ranks = current_ranks

                print(f"Advanced to Round {gs.round_idx}. New pairs added: {len(new_round_pairings)}")
                await save_state()
                # Try to fill all idle tables now that queue has new pairs
                for i in range(MAX_TABLES):
                    if i not in gs.active_matches:
                        # Call non-blockingly, attempt_to_seat_next_match will check table availability
                        asyncio.create_task(attempt_to_seat_next_match(i))
            else:
                print(f"No new pairings generated for round {gs.round_idx + 1}.")
                # This could mean tournament is over due to no possible pairs or everyone played everyone

        if not gs.match_queue: # Still no matches in queue after trying to generate
            update_table_card_ui_for_idle(table_idx, "Ожидаем следующий раунд")
            return

    # Try to find a (P1, P2) pair first from the queue
    # This prioritizes actual matches over players waiting for a floater (P1, None)
    pair_to_seat_idx = -1
    is_floater_request = False

    for i, (p1, p2_opt) in enumerate(gs.match_queue):
        if p1 and p2_opt: # This is a (P1, P2) pair
            # Simple check: are players available (not in active_matches)?
            # More robust check would be to ensure they are not player_a of a (player_a, None)
            # that is also in queue, but choose_floater logic should handle that.
            # For now, assume if they are in queue as (P1,P2), they are intended to play.
            pair_to_seat_idx = i
            is_floater_request = False
            break

    if pair_to_seat_idx == -1: # No (P1, P2) pairs, look for (P1, None)
        for i, (p1, p2_opt) in enumerate(gs.match_queue):
            if p1 and p2_opt is None: # This is a (P1, None) request
                pair_to_seat_idx = i
                is_floater_request = True
                break

    if pair_to_seat_idx == -1: # Queue has items, but none are suitable (e.g. all (P1,P2) repeats that swiss couldn't avoid?)
                               # Or queue is empty after all.
        update_table_card_ui_for_idle(table_idx, "Нет доступных пар в очереди")
        return

    # Pop the chosen pair/request from queue
    p1_name_raw, p2_name_opt_raw = gs.match_queue.pop(pair_to_seat_idx)

    final_p1_name: str
    final_p2_name: Optional[str]

    if is_floater_request: # It was (p1_name_raw, None)
        final_p1_name = p1_name_raw
        final_p2_name = choose_floater_for(final_p1_name)
        if final_p2_name is None:
            # No floater found, put P1 back in queue to wait.
            gs.match_queue.append((final_p1_name, None)) # Add to end of queue
            update_table_card_ui_for_idle(table_idx, f"{final_p1_name} ожидает партнера")
            return
    else: # It was (p1_name_raw, p2_name_opt_raw as P2)
        final_p1_name = p1_name_raw
        final_p2_name = p2_name_opt_raw # Should be a string name here

    if final_p2_name is None: # Should not happen if logic is correct above
        print(f"Error: Attempted to seat match for {final_p1_name} but P2 is None unexpectedly.")
        update_table_card_ui_for_idle(table_idx, "Ошибка формирования пары")
        # Re-add p1_name_raw if it was from a (P1,P2) that somehow lost P2. Unlikely.
        return

    # At this point, final_p1_name and final_p2_name are set for a match.
    # Redundancy: Check for forbidden pairs IF this pair is also a repeat.
    # Normal Swiss/Floater logic tries to avoid forbidden, then repeats.
    # A repeat forbidden pair is the worst case and should be logged if chosen by necessity.
    current_match_players_set = frozenset({final_p1_name, final_p2_name})
    is_repeat = current_match_players_set in gs.played_pairs
    is_forbidden = current_match_players_set in FORBIDDEN_PAIRS

    if is_repeat and is_forbidden:
        print(f"CRITICAL SEATING WARNING: Seating REPEAT and FORBIDDEN pair: {final_p1_name} vs {final_p2_name} on table {table_idx}.")
        # Proceeding as this might be a last resort by floater/swiss logic

    p_red_name, p_blue_name = fair_break(final_p1_name, final_p2_name)
    table_display_name = gs.table_names[table_idx]
    match = Match(p_red=p_red_name, p_blue=p_blue_name, table_name=table_display_name, table_idx=table_idx)
    match.start_time = datetime.datetime.now()
    match.match_timer_start_monotonic = time.monotonic()

    gs.active_matches[table_idx] = match
    gs.players[final_p1_name].played_with.add(final_p2_name)
    gs.players[final_p2_name].played_with.add(final_p1_name)
    gs.played_pairs.add(current_match_players_set) # Add to played_pairs

    print(f"Seating match on {table_display_name} (Idx {table_idx}): {p_red_name} (Red) vs {p_blue_name} (Blue)")
    update_table_card_ui_for_match(table_idx, match)
    await save_state()


async def handle_match_result(table_idx: int, winner_name: str):
    global gs
    if table_idx not in gs.active_matches:
        return

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
    if duration_sec is not None:
        winner_obj.win_durations.append(duration_sec)
    loser_obj.losses += 1
    timestamp_now = time.time()
    winner_obj.last_played_timestamp = timestamp_now
    loser_obj.last_played_timestamp = timestamp_now

    update_scoreboard_display()
    clear_table_card_ui(table_idx)
    await save_state()
    await attempt_to_seat_next_match(table_idx) # Try to seat next match on this now free table
    await show_finish_screen_if_needed() # Check if tournament end conditions met

#--- UI Rendering & Updates ---
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
            top_bar_tournament_time_label.set_text(f"До конца: {str(remaining_time).split('.')[0]}")
            top_bar_tournament_time_label.style(remove="color: red;")
        else:
            over_time = now - TOURNAMENT_END_DATETIME
            top_bar_tournament_time_label.set_text(f"Время вышло: -{str(over_time).split('.')[0]}")
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
    if not card_elements: return

    p_red_btn_el = card_elements.get('p_red_btn')
    p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label')
    timer_label_el = card_elements.get('timer_label')
    card_el = card_elements.get('card')

    if not all([p_red_btn_el, p_blue_btn_el, status_label_el, timer_label_el, card_el]): return

    p_red_btn_el.set_text(match.p_red)
    p_red_btn_el.props('disable=false flat=false unelevated=false').classes(remove='button-idle', add='player-red')

    p_blue_btn_el.set_text(match.p_blue)
    p_blue_btn_el.props('disable=false flat=false unelevated=false').classes(remove='button-idle', add='player-blue')

    status_label_el.set_text(match.table_name)
    status_label_el.classes(remove='text-grey')
    timer_label_el.set_text("00:00") # Will be updated by update_match_timers
    card_el.classes(remove='table-card-idle')

def clear_table_card_ui(table_idx: int):
    update_table_card_ui_for_idle(table_idx, "Ожидаем пару...")

def update_table_card_ui_for_idle(table_idx: int, message: str):
    card_elements = table_cards_display.get(table_idx)
    if not card_elements: return

    p_red_btn_el = card_elements.get('p_red_btn')
    p_blue_btn_el = card_elements.get('p_blue_btn')
    status_label_el = card_elements.get('status_label')
    timer_label_el = card_elements.get('timer_label')
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
    def sort_key_scoreboard(p: Player):
        awt = p.average_win_time()
        sb_score = p.sonnenborn_berger_score(gs.players)
        return (-p.wins, awt if awt is not None else float('inf'), -sb_score)

    present_players_list.sort(key=sort_key_scoreboard)

    # Assign ranks
    ranks = {}
    for i, p_obj in enumerate(present_players_list):
        key_curr = sort_key_scoreboard(p_obj)
        if i == 0:
            ranks[p_obj.name] = 1
        else:
            prev_p_obj = present_players_list[i-1]
            key_prev = sort_key_scoreboard(prev_p_obj)
            if key_curr == key_prev: # Tie in sort keys means same rank
                ranks[p_obj.name] = ranks[prev_p_obj.name]
            else:
                ranks[p_obj.name] = i + 1 # Rank is current position (1-indexed)
        p_obj.current_rank = ranks[p_obj.name]
    return present_players_list

def update_scoreboard_display(target_container: Optional[ui.html] = None):
    global scoreboard_show_details # Use the global toggle state

    current_scoreboard_container = target_container if target_container else scoreboard_container

    if not current_scoreboard_container or (not gs.tournament_started and not gs.tournament_finish_screen_shown):
        return

    present_players_list = _get_sorted_players_for_scoreboard()

    if scoreboard_show_details:
        html = "<table class='score-table'><thead><tr><th>#</th><th>Имя</th><th class='score-value'>Побед</th><th class='score-value'>Ср. время</th><th class='score-value'>СБ</th></tr></thead><tbody>"
    else:
        html = "<table class='score-table'><thead><tr><th>#</th><th>Имя</th><th class='score-value'>Побед</th></tr></thead><tbody>"

    if not present_players_list:
        cols = 5 if scoreboard_show_details else 3
        html += f"<tr><td colspan='{cols}' style='text-align:center; padding: 20px;'>Нет игроков для отображения</td></tr>"
    else:
        for p_obj in present_players_list:
            if scoreboard_show_details:
                avg_win_time_str = format_duration(p_obj.average_win_time()) if p_obj.average_win_time() is not None else "-"
                sb_str = f"{p_obj.sonnenborn_berger_score(gs.players):.2f}"
                html += f"<tr><td>{p_obj.current_rank}</td><td class='player-name-cell'>{p_obj.name}</td><td class='score-value'>{p_obj.wins}</td><td class='score-value'>{avg_win_time_str}</td><td class='score-value'>{sb_str}</td></tr>"
            else:
                html += f"<tr><td>{p_obj.current_rank}</td><td class='player-name-cell'>{p_obj.name}</td><td class='score-value'>{p_obj.wins}</td></tr>"
    html += "</tbody></table>"
    current_scoreboard_container.set_content(html)


async def show_confirmation_dialog(table_idx: int, player_name: str):
    try:
        table_display_name = f"Стол {table_idx + 1}"
        if 0 <= table_idx < len(gs.table_names):
            table_display_name = gs.table_names[table_idx]

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Победил {player_name} на столе {table_display_name}?")
            with ui.row().classes("justify-end w-full"):
                ui.button("Нет", on_click=lambda: dialog.submit(False), color='negative')
                ui.button("Да", on_click=lambda: dialog.submit(True), color='positive')

        result = await dialog
        if result:
            await handle_match_result(table_idx, player_name)
        else:
            print(f"INFO: Match result recording cancelled for {player_name} on table {table_idx}.")

    except RuntimeError as e: # Specific to NiceGUI UI creation issues in some contexts
        print(f"FATAL ERROR in show_confirmation_dialog creating UI: {e}")
        ui.notify(f"Ошибка отображения диалога: {e}", type='negative', multi_line=True, close_button=True)
    except Exception as e: # Catch any other unexpected errors
        print(f"UNEXPECTED ERROR in show_confirmation_dialog: {e}")
        ui.notify(f"Неожиданная ошибка: {e}", type='negative', multi_line=True, close_button=True)


async def show_finish_screen_if_needed():
    global gs, ui_container, finish_screen_scoreboard_container
    tournament_time_over = datetime.datetime.now() >= TOURNAMENT_END_DATETIME
    no_active_matches = not gs.active_matches

    if tournament_time_over and no_active_matches and gs.tournament_started and not gs.tournament_finish_screen_shown:
        gs.tournament_finish_screen_shown = True
        await save_state()

        if ui_container is None: return

        ui_container.clear()
        with ui_container:
            with ui.splitter(value=50).classes("w-full h-full") as finish_splitter:
                with finish_splitter.before: # Left panel for podium and stats
                    with ui.column().classes("w-full h-full items-center q-pa-md overflow-auto"): # No scroll
                        ui.label("Турнир Завершен!").classes("text-h3 self-center q-my-md")

                        present_players_list = _get_sorted_players_for_scoreboard()
                        final_ranks_map = {p.name: p.current_rank for p in present_players_list}

                        # Podium
                        with ui.row().classes("justify-center q-gutter-md q-my-lg"):
                            if len(present_players_list) >= 1:
                                p1 = present_players_list[0]
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥇").classes("podium-1")
                                    ui.label(f"{p1.name}").classes("podium-player-name")
                                    ui.label(f"{p1.wins} побед").classes("podium-player-wins")
                            
                            p2 = None
                            if len(present_players_list) >= 2:
                                p2 = next((p for p in present_players_list if final_ranks_map.get(p.name) == 2), None)
                                if not p2 and final_ranks_map.get(present_players_list[1].name) != 1:
                                     p2 = present_players_list[1]
                            if p2:
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥈").classes("podium-2")
                                    ui.label(f"{p2.name}").classes("podium-player-name")
                                    ui.label(f"{p2.wins} побед").classes("podium-player-wins")

                            p3 = None
                            if len(present_players_list) >=3:
                                p3 = next((p for p in present_players_list if final_ranks_map.get(p.name) == 3), None)
                                if not p3:
                                    distinct_ranks_seen = 0
                                    last_rank_seen = 0
                                    for p_candidate in present_players_list:
                                        current_cand_rank = final_ranks_map.get(p_candidate.name,0)
                                        if current_cand_rank != last_rank_seen:
                                            distinct_ranks_seen +=1
                                            last_rank_seen = current_cand_rank
                                        if distinct_ranks_seen == 3:
                                            p3 = p_candidate
                                            break
                            if p3:
                                with ui.card().classes("podium-card items-center"):
                                    ui.label("🥉").classes("podium-3")
                                    ui.label(f"{p3.name}").classes("podium-player-name")
                                    ui.label(f"{p3.wins} побед").classes("podium-player-wins")

                        with ui.card().classes("q-my-md w-full"):
                            ui.label("Статистика турнира:").classes("text-h6 q-mb-sm")
                            if gs.finished_matches:
                                ui.label(f"Всего сыграно матчей: {len(gs.finished_matches)}")
                                valid_durations = [m.duration_seconds() for m in gs.finished_matches if m.duration_seconds() is not None and m.duration_seconds() > 0]
                                if valid_durations:
                                    avg_duration = sum(valid_durations) / len(valid_durations)
                                    ui.label(f"Средняя длительность матча: {format_duration(avg_duration)}")
                                fastest_match = min(gs.finished_matches, key=lambda m: m.duration_seconds() or float('inf'))
                                fm_duration = fastest_match.duration_seconds()
                                if fm_duration is not None:
                                    ui.label(f"Самый быстрый матч: {fastest_match.p_red} vs {fastest_match.p_blue} ({format_duration(fm_duration)})")
                                longest_match = max(gs.finished_matches, key=lambda m: m.duration_seconds() or float('-inf'))
                                lm_duration = longest_match.duration_seconds()
                                if lm_duration is not None:
                                    ui.label(f"Самый долгий матч: {longest_match.p_red} vs {longest_match.p_blue} ({format_duration(lm_duration)})")

                            # Highest Win Rate
                            min_games_for_win_rate = 3
                            best_win_rate_players_names = []
                            highest_win_rate = -1.0
                            for p_obj_stat in get_present_players(): # Iterate all present players
                                total_games = p_obj_stat.wins + p_obj_stat.losses
                                if total_games >= min_games_for_win_rate:
                                    win_rate = p_obj_stat.wins / total_games
                                    if win_rate > highest_win_rate:
                                        highest_win_rate = win_rate
                                        best_win_rate_players_names = [p_obj_stat.name]
                                    elif win_rate == highest_win_rate:
                                        best_win_rate_players_names.append(p_obj_stat.name)
                            if best_win_rate_players_names:
                                ui.label(f"Лучшая доля побед (мин. {min_games_for_win_rate} игр): {', '.join(best_win_rate_players_names)} ({highest_win_rate:.0%})")

                            # Most Unique Opponents
                            most_unique_opp_players_names = []
                            max_unique_opponents = -1
                            for p_obj_stat in get_present_players():
                                num_unique = len(p_obj_stat.played_with)
                                if num_unique > max_unique_opponents:
                                    max_unique_opponents = num_unique
                                    most_unique_opp_players_names = [p_obj_stat.name]
                                elif num_unique == max_unique_opponents:
                                     most_unique_opp_players_names.append(p_obj_stat.name)
                            if most_unique_opp_players_names and max_unique_opponents > 0 :
                                ui.label(f"Сыграл(и) с макс. числом разных оппонентов: {', '.join(most_unique_opp_players_names)} ({max_unique_opponents})")

                            # Best Comeback (Initial Rank vs Final Rank)
                            best_comeback_player_name = None
                            max_rank_improvement = -float('inf') # Can be negative if rank worsened
                            if gs.initial_ranks:
                                for p_obj_cb in present_players_list: # Use sorted list for final ranks
                                    if p_obj_cb.name in gs.initial_ranks:
                                        initial_rank = gs.initial_ranks[p_obj_cb.name]
                                        final_rank = final_ranks_map.get(p_obj_cb.name, initial_rank) # current_rank from sorted list
                                        improvement = initial_rank - final_rank # Higher number is better (e.g. rank 10 to 1 is +9)
                                        if improvement > max_rank_improvement :
                                            max_rank_improvement = improvement
                                            best_comeback_player_name = p_obj_cb.name
                                if best_comeback_player_name and max_rank_improvement > 0:
                                    initial_r = gs.initial_ranks[best_comeback_player_name]
                                    final_r = final_ranks_map.get(best_comeback_player_name)
                                    ui.label(f"Лучший камбэк: {best_comeback_player_name} (с {initial_r} на {final_r} место, +{max_rank_improvement} поз.)")
                
                with finish_splitter.after: # Right panel for final scoreboard
                    with ui.column().classes("w-full h-full items-center q-pa-md overflow-auto"): # No scroll
                        ui.label("Итоговая Таблица Лидеров").classes("text-h5 section-title q-mb-sm")
                        finish_screen_scoreboard_container = ui.html().classes("w-full")
                        # For finish screen, always show details
                        global scoreboard_show_details
                        original_show_details_temp = scoreboard_show_details
                        scoreboard_show_details = True # Force detailed view for final scoreboard
                        update_scoreboard_display(target_container=finish_screen_scoreboard_container)
                        scoreboard_show_details = original_show_details_temp # Restore global setting

def build_setup_ui(container: ui.column):
    container.clear()

    async def start_tournament_action():
        global gs, ui_container
        gs.tournament_started = True
        gs.round_idx = 0 # Initial pairings are not a "round" yet
        gs.current_round_start_time = datetime.datetime.now() # Mark start time
        gs.initial_ranks = {} # Reset initial ranks
        gs.match_queue = generate_initial_pairings()

        # Populate initial ranks based on current state (all 0 wins)
        update_scoreboard_display() # This calculates current_rank for all players
        current_ranks = {p.name: p.current_rank for p in get_present_players()}
        gs.initial_ranks = current_ranks

        await save_state()
        if ui_container:
            build_dashboard_ui(ui_container)
            # Attempt to seat initial matches
            for table_idx in range(MAX_TABLES):
                if table_idx not in gs.active_matches:
                    await attempt_to_seat_next_match(table_idx)
            update_scoreboard_display() # Refresh scoreboard after potential seating

    with container.classes("overflow-auto"): # Prevent scroll on setup screen too
        ui.label("Настройка Турнира").classes("text-h4 self-center q-mb-md section-title")
        with ui.card().classes("w-full"):
            ui.label("Участники:").classes("text-h6")
            present_count_ui_label = ui.label().classes("q-mb-sm")
            # Grid for checkboxes, ensure it doesn't cause overflow if possible
            checkbox_grid = ui.grid(columns=3).classes("q-gutter-sm w-full")
        with ui.card().classes("w-full q-mt-md"):
            ui.label("Столы:").classes("text-h6")
            for i in range(MAX_TABLES):
                while len(gs.table_names) <= i: # Should not happen if MAX_TABLES is respected by gs.table_names init
                    gs.table_names.append(f"Стол {len(gs.table_names) + 1}")
                ui.input(f"Название стола {i+1}", value=gs.table_names[i],
                         on_change=lambda e, idx=i: gs.table_names.__setitem__(idx, e.value))

        the_start_button = ui.button("Начать турнир", on_click=start_tournament_action)
        the_start_button.props("color=primary").classes("q-mt-lg self-center")

        def update_presence_and_button_state():
            count = sum(1 for p_name in PLAYERS_PRESET if gs.players[p_name].present)
            present_count_ui_label.set_text(f"Присутствуют: {count} из {len(PLAYERS_PRESET)}")
            the_start_button.props(f"disable={count < 2}") # Need at least 2 players

        with checkbox_grid:
            for name_key in PLAYERS_PRESET:
                if name_key not in gs.players: # Should be pre-populated by GameState init
                    gs.players[name_key] = Player(name_key)
                player_obj = gs.players[name_key]
                ui.checkbox(name_key, value=player_obj.present,
                            on_change=lambda e, p=player_obj: (setattr(p, 'present', e.value), update_presence_and_button_state()))
        update_presence_and_button_state() # Initial call


def build_dashboard_ui(container: ui.column):
    container.clear()
    global top_bar_round_label, top_bar_round_time_label, top_bar_tournament_time_label
    global table_cards_display, scoreboard_container, toggle_scoreboard_btn_ref

    with container:
        # Top bar for timers and round info
        with ui.row().classes("w-full justify-between items-center q-pa-sm bg-grey-9 text-white q-mb-md"):
            top_bar_round_label = ui.label("Раунд N")
            top_bar_round_time_label = ui.label("⏱ Раунда: 00:00").classes("timer-display")
            top_bar_tournament_time_label = ui.label("До конца: HH:MM:SS").classes("timer-display")

        # Main content splitter (Tables | Scoreboard)
        with ui.splitter(value=60).classes("w-full h-full dashboard-splitter") as splitter:
            with splitter.before: # Tables Panel
                with ui.column().classes("w-full h-full overflow-auto"): # No scroll for table grid area
                    with ui.grid(columns=2 if MAX_TABLES > 1 else 1).classes("w-full q-gutter-md"):
                        for i in range(MAX_TABLES):
                            table_name = gs.table_names[i]
                            # Lambdas capture 'i' correctly by making it a default argument
                            async def handle_red_click(captured_table_idx=i):
                                match = gs.active_matches.get(captured_table_idx)
                                if match and match.p_red:
                                    await show_confirmation_dialog(captured_table_idx, match.p_red)
                            async def handle_blue_click(captured_table_idx=i):
                                match = gs.active_matches.get(captured_table_idx)
                                if match and match.p_blue:
                                    await show_confirmation_dialog(captured_table_idx, match.p_blue)

                            with ui.card().classes("table-card table-card-idle items-center justify-between") as local_card_element:
                                status_label = ui.label(f"{table_name} - Ожидаем...").classes("text-subtitle1 q-mb-sm text-center")
                                p_red_btn = ui.button("---", color="red", on_click=handle_red_click) \
                                    .props('flat unelevated disable=true').classes('button-idle full-width q-my-xs')
                                p_blue_btn = ui.button("---", color="blue", on_click=handle_blue_click) \
                                    .props('flat unelevated disable=true').classes('button-idle full-width q-my-xs')
                                timer_label = ui.label("--:--").classes("timer-display self-center q-mt-sm")

                                table_cards_display[i] = {
                                    'card': local_card_element, 'status_label': status_label,
                                    'p_red_btn': p_red_btn, 'p_blue_btn': p_blue_btn, 'timer_label': timer_label
                                }
                                # Restore state if match is active on this table
                                if i in gs.active_matches:
                                    update_table_card_ui_for_match(i, gs.active_matches[i])
                                else:
                                    clear_table_card_ui(i) # Sets to idle state

            with splitter.after: # Scoreboard Panel
                with ui.column().classes("w-full h-full items-center q-pa-xs overflow-auto"): # No scroll for scoreboard area
                    with ui.row().classes("w-full justify-between items-center"):
                         ui.label("Таблица Лидеров").classes("text-h5 section-title q-mb-sm")
                         toggle_scoreboard_btn_ref = ui.button(
                             icon='sym_o_visibility_off' if scoreboard_show_details else 'sym_o_visibility',
                             on_click=toggle_scoreboard_details
                         ).props("flat dense")
                         toggle_scoreboard_btn_ref.tooltip("Показать/Скрыть детали таблицы")
                    scoreboard_container = ui.html().classes("w-full") # Scoreboard HTML will be injected here

    update_top_bar_timers()
    update_scoreboard_display()

#--- Main App Setup & Page Definition ---
@ui.page('/')
async def main_page(client: Client):
    global ui_container, gs

    ui.dark_mode().enable()
    # Main container takes full screen height and uses flex to manage its direct children
    ui_container = ui.column().classes("w-full h-screen items-stretch q-pa-md main-container")

    if gs.tournament_finish_screen_shown:
        await show_finish_screen_if_needed() # Rebuilds the finish screen
    elif gs.tournament_started:
        build_dashboard_ui(ui_container)
        # After dashboard UI is built, ensure tables reflect current match states or try to seat new ones
        for table_idx in range(MAX_TABLES):
            if table_idx in gs.active_matches:
                # update_table_card_ui_for_match is called inside build_dashboard_ui
                pass
            else: # If table is idle, try to seat a match
                await attempt_to_seat_next_match(table_idx)
        update_scoreboard_display() # Initial scoreboard render
        # Ensure initial ranks are captured if loading into an ongoing tournament without them
        if gs.round_idx >= 1 and not gs.initial_ranks:
            current_ranks = {p.name: p.current_rank for p in get_present_players()}
            gs.initial_ranks = current_ranks
            await save_state()
    else:
        build_setup_ui(ui_container) # Tournament not started, show setup

    # Timers for dynamic updates
    ui.timer(1.0, update_top_bar_timers)
    ui.timer(0.5, update_match_timers) # More frequent for match timers
    # Timer to check for tournament end condition
    ui.timer(5.0, lambda: asyncio.create_task(show_finish_screen_if_needed()))


#--- AutoSave Loop ---
async def auto_save_loop():
    while True:
        await asyncio.sleep(AUTOSAVE_INTERVAL)
        if gs.tournament_started and not gs.tournament_finish_screen_shown : # Only save if tournament is active
            await save_state()
            print(f"Autosaved state at {datetime.datetime.now()}")

#--- App Startup Hook ---
@app.on_startup
async def on_app_startup():
    load_state_sync()
    # scoreboard_show_details is global, will be its last value or default False.
    # If it were part of saved state, you'd load it here.
    # gs.scoreboard_show_details = loaded_settings.get("scoreboard_details", False)
    if toggle_scoreboard_btn_ref: # This ref might not be set yet on initial startup before page build
         pass # Button icon will be set during build_dashboard_ui or main_page logic
    asyncio.create_task(auto_save_loop())

#--- App Shutdown Hook ---
@app.on_shutdown
async def on_app_shutdown():
    if gs.tournament_started : # Save one last time on shutdown if tournament was running
        print("Shutdown: Saving final state...")
        save_state_sync()

#--- CSS Styles ---
ui.add_head_html(f'''
<style>
html, body {{ height: 100%; overflow: hidden; }} /* Global no-scroll */
body {{ background:#121212; color:#e0e0e0; font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}

.main-container {{
    height: 100vh; max-height: 100vh; /* Fill viewport */
    display: flex; flex-direction: column; /* Allow children to take height properly */
    overflow: hidden; /* Ensure main container itself does not scroll */
}}
.dashboard-splitter.q-splitter {{ height: calc(100% - 50px); /* Adjust based on top bar height if fixed */ }}
.dashboard-splitter > .q-splitter__before,
.dashboard-splitter > .q-splitter__after {{
    overflow: hidden !important; /* Prevent scrollbars on splitter panels */
}}
.q-splitter__panel {{ overflow: hidden !important; }} /* General rule for panels inside splitter */

.q-btn {{ font-size:1.0rem !important; padding:0.5rem 0.8rem !important; border-radius:6px !important; text-transform: none !important; line-height: 1.2 !important; min-height: 40px !important;}}
.q-btn.player-red {{ background:#b71c1c !important; color:#fff !important; }}
.q-btn.player-blue {{ background:#0d47a1 !important; color:#fff !important; }}
.q-btn.button-idle {{ background-color: #333 !important; color: #666 !important; box-shadow: none !important; border: 1px solid #444 !important; }}
.q-card {{ border:1px solid #333; background-color: #1e1e1e; }}
.timer-display {{ font-family: "Consolas", "Monaco", monospace; font-size: 1.2rem; }}
.section-title {{ font-size: 1.3rem; font-weight: bold; margin-bottom: 8px; color: #bbb; }}

.table-card {{ min-height: 190px; padding: 10px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; }}
.table-card-idle {{ background-color: #282828 !important; color: #777 !important; }}
.table-card .q-btn {{ white-space: normal; word-break: break-word; }} /* Allow button text to wrap */

/* Scoreboard Styling */
.score-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 1.0rem; /* Base font for scoreboard */
}}
.score-table th, .score-table td {{
    padding: 7px 10px; /* Padding for cells */
    text-align: left;
    border-bottom: 1px solid #383838; /* Row separator */
    line-height: 1.4; /* Improve readability */
}}
.score-table th {{
    background-color: #2a2a2a;
    font-weight: bold;
    /* position: sticky; top: 0; */ /* Only if scoreboard area itself scrolls */
}}
.score-table .player-name-cell {{
    /* font-weight: 500; */ /* Optional: bolder player names */
}}
.score-table .score-value {{
    text-align: right; /* Right-align scores */
    font-family: "Consolas", "Monaco", monospace; /* Monospaced for alignment */
    min-width: 50px; /* Ensure some space for scores */
}}
.score-table tbody tr:hover {{
    background-color: #2c2c2c; /* Hover effect for rows */
}}

/* Podium Styling */
.podium-card {{ padding: 15px; text-align: center; background-color: #222; min-width: 130px; }}
.podium-1 {{ font-size: 2.8rem; color: gold; }} /* Emoji for 1st */
.podium-2 {{ font-size: 2.2rem; color: silver; }} /* Emoji for 2nd */
.podium-3 {{ font-size: 1.9rem; color: #cd7f32; }} /* Emoji for 3rd */

.podium-player-name {{
    font-size: 1.4rem; /* Consistent for all podium names */
    font-weight: bold;
    margin-top: 4px;
    margin-bottom: 3px;
    line-height: 1.2;
}}
.podium-player-wins {{
    font-size: 0.95rem; /* Consistent for all podium win counts */
    color: #b0b0b0;
}}

.text-grey {{ color: #777 !important; }}
.q-checkbox__inner {{ width: 22px !important; height: 22px !important; min-width:22px !important;}}
.q-checkbox__bg {{ width: 100% !important; height: 100% !important;}}
.q-checkbox__label {{ font-size: 1.0rem !important; padding-left: 6px;}}
.q-input .q-field__native {{ font-size: 1.0rem !important; }}
.q-input .q-field__label {{ font-size: 1.0rem !important; }}

.overflow-auto {{ overflow: hidden !important; }} /* Utility class if needed beyond direct styling */
.h-full {{ height: 100%; }}
.w-full {{ width: 100%; }}
</style>
''')

ui.run(title="Billiard Tournament Console", port=8080, reload=False, native=True, fullscreen=True, window_size=(1280,800))