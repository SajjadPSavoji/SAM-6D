# install rclone
sudo apt install rclone

# Open following link with your google accont. https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing
# The dataset will be added to your drive under "shared with me"/"train_data"

# configure rclone for google drive
# Run the configuration command:
# rclone config
# Create a new remote:
# Type n and press Enter.
# When prompted, give your remote a name (for example, gdrive).
# Select Google Drive as the storage type:
# From the list, type the number corresponding to Google Drive (usually 13 or similar) and press Enter.
# Client ID and Secret:
# You can leave these blank to use rclone's default, or follow the Google Drive API instructions to create your own credentials.
# Authentication:
# Choose auto config (usually typing y) which will open a browser window.
# Log in with your Google account and allow rclone to access your Google Drive.
# Advanced config:
# You can typically leave the advanced configuration blank by pressing Enter for defaults.
# Confirm the configuration:
# Once complete, your new remote will be saved in the rclone config file.
rclone config

# list the files shared with you
rclone lsd gdrive: --drive-shared-with-me


# download dataset to local
rclone copy "gdrive:train_data" --drive-shared-with-me --progress \
  --transfers 8 --checkers 16 --buffer-size 64M --multi-thread-streams 4 --fast-list

