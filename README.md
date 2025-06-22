# Billiard Tournament Console

## Overview

The Billiard Tournament Console is a Python application built with [NiceGUI](https://nicegui.io/) to manage and display the progress of billiard tournaments. It facilitates player management, automates match pairings using a Swiss-like system, tracks scores in real-time, and provides a comprehensive dashboard for participants and organizers. The application persists its state, allowing for tournaments to be paused and resumed.

## Features

*   **Player Management:**
    *   Uses a preset list of players.
    *   Allows marking players as present or absent for the tournament.
*   **Automated Match Pairing:**
    *   Generates initial pairings randomly.
    *   Subsequent rounds use a Swiss-like system, pairing players with similar win records.
    *   Considers "forbidden pairs" to avoid specific player matchups if possible.
    *   Handles floaters (players without an opponent in a round) by attempting to find suitable partners.
*   **Real-Time Dashboard:**
    *   Displays active matches on available tables with live timers.
    *   Shows a dynamic scoreboard with player rankings based on:
        *   Number of wins.
        *   Average win time.
        *   Sonnenborn-Berger coefficient.
    *   Indicates current round number and time elapsed in the round.
    *   Features a countdown timer to the scheduled end of the tournament.
*   **Match Result Confirmation:**
    *   Requires user confirmation for reported match winners.
*   **State Persistence:**
    *   Saves the entire tournament state (players, matches, scores, round progress) to `state.json`.
    *   Includes an auto-save feature to prevent data loss.
*   **Tournament Statistics:**
    *   Calculates and displays various statistics upon tournament completion, such as:
        *   Total matches played.
        *   Average match duration.
        *   Fastest and longest matches.
        *   Player(s) who played the most games.
        *   Players who participated as floaters.
        *   "Best Comeback" player based on rank improvement.
*   **Final Results Screen:**
    *   Presents a podium for the top players (1st, 2nd, 3rd).
    *   Shows the final detailed leaderboard.
*   **Customizable Table Names:**
    *   Allows setting custom names for each billiard table.

## How to Run

1.  **Prerequisites:**
    *   Python 3.7+
    *   Install NiceGUI and its dependencies. Typically, this can be done via pip:
        ```bash
        pip install nicegui
        ```

2.  **Running the Application:**
    *   Navigate to the project directory in your terminal.
    *   Execute the main script:
        ```bash
        python main.py
        ```
    *   The application will start a web server, and you can access the console by opening a web browser to `http://localhost:8080` (or the address shown in the terminal output).
    *   The application is designed to run in fullscreen (native mode) by default.

## Project Structure

*   `main.py`: Contains all the application logic, including UI definitions (using NiceGUI), tournament management algorithms, state handling, and embedded CSS for styling.
*   `state.json`: Automatically generated file that stores the current state of the tournament. This includes player data, active and finished matches, current round, queue, and other relevant information. It allows the tournament to be resumed if the application is closed.

## Key Configuration Constants

Several constants at the beginning of `main.py` can be modified to customize the tournament:

*   `PLAYERS_PRESET`: A list of strings defining the default set of player names.
*   `FORBIDDEN_PAIRS_NAMES`: A list of tuples, where each tuple contains two player names that should ideally not be paired against each other.
*   `MAX_TABLES`: An integer specifying the number of billiard tables available for matches.
*   `TOURNAMENT_END_DATETIME`: A `datetime` object indicating the scheduled end time for the tournament.
*   `AUTOSAVE_INTERVAL`: An integer representing the interval in seconds at which the tournament state is automatically saved to `state.json`.

## Styling

The application's visual appearance is defined by CSS styles embedded directly within the `main.py` script using `ui.add_head_html()`. This includes styling for table cards, scoreboard, buttons, and overall layout.
