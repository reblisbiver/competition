# Choice Engineering Competition - Experimental Platform

## Overview
This is a PHP-based web application for the Choice Engineering Competition, an academic competition that invites participants to devise mechanisms that engineer behavior. The application runs a psychological experiment where subjects make repeated choices between two buttons to collect rewards (smiley faces).

## Project Structure

### Main Components
- **experiment/main.php** - Main experiment page that manages the game flow and UI
- **experiment/scripts/backend.php** - Backend logic for reward allocation and data logging
- **experiment/scripts/code.js** - Frontend game logic and UI interactions
- **experiment/scripts/backend_interaction.js** - AJAX communication between frontend and backend
- **experiment/sequences/static/** - Static reward schedules (PHP arrays)
- **experiment/sequences/dynamic/** - Dynamic reward schedules (Python scripts)
- **experiment/scripts/results/** - Directory where experiment data is saved as CSV files

### Data Storage
- Data files (CSV and JSON) are stored in `data/` and `data_unfiltered/` directories
- New experiment results are written to `experiment/scripts/results/`
- Results are organized by schedule type (STATIC/DYNAMIC) and schedule name

## How It Works

### Experiment Flow
1. Subject sees instructions (welcome.html)
2. Main experiment begins with 100 trials
3. Subject chooses between left/right buttons (randomly colored red or blue)
4. Backend allocates rewards based on the selected schedule
5. Feedback is displayed (smiley face for reward, sad face for no reward)
6. Results are logged to CSV files
7. After 100 trials, experiment ends with thank you page

### Reward Schedules
- **STATIC**: Pre-defined reward sequences loaded from PHP files (e.g., random_0.php)
- **DYNAMIC**: Generated on-the-fly using Python scripts that adapt based on subject choices

### Session Management
PHP sessions track:
- User ID (timestamp_randomNumber)
- Current trial number
- Total rewards collected
- Choice history (is_bias_choice array)
- Reward history (bias_rewards and unbias_rewards arrays)
- Biased/unbiased side assignment (LEFT/RIGHT randomly assigned)

## Running the Application

The application runs on PHP 8.2 using the built-in development server on port 5000.

### Development
The workflow "PHP Web Server" automatically starts the server with:
```
php -S 0.0.0.0:5000 -t experiment
```

### Accessing the Experiment
- Navigate to the root URL to automatically redirect to main.php
- Or access directly via `/main.php`

## Configuration

### Changing the Reward Schedule
Edit `experiment/main.php`:
```php
$_SESSION['schedule_type'] = $TYPE_STATIC; // or $TYPE_DYNAMIC
$_SESSION['schedule_name'] = "random_0"; // name of schedule file
```

### Creating New Static Schedules
1. Create a new PHP file in `experiment/sequences/static/`
2. Define two arrays: `$biased_rewards` and `$unbiased_rewards`
3. Each array should have 100 elements (0 or 1)

### Creating New Dynamic Schedules
1. Create a Python script in `experiment/sequences/dynamic/`
2. Script receives: bias_rewards history, unbias_rewards history, choice history, user_id
3. Script should output: "biased_reward, unbiased_reward"

## Recent Changes
- 2025-11-26: Set up for Replit environment
  - Installed PHP 8.2 module
  - Created index.php for automatic redirect to main.php
  - Configured workflow to run PHP server on port 5000
  - Created results directory structure
  - Updated .gitignore to exclude results and CSV files
  - Configured deployment for VM target

## External Resources
- [Competition Website](http://decision-making-lab.com/competition/index.html)
- Original repository: https://github.com/ohaddan/competition

## Notes
- The application uses PHP sessions, so each user gets a unique experience
- LSP warnings in backend.php about undefined variables are false positives (variables defined in included files)
- Results are automatically saved to CSV files with trial-by-trial data
