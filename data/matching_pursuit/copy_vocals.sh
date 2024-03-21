# Define the source and destination directories
SRC_DIR="/home/louis/Documents/notebooks/spleeter/taylor/separated/htdemucs"
DEST_DIR="taylor_vocals"

# Initialize a counter
counter=1

# Find all CSV files in the source directory and its subdirectories
find "$SRC_DIR" -type f -name "*.csv" | while read file; do
    # Extract the filename without the path
    filename=$(basename -- "$file")
    # Copy the file to the destination directory with the counter appended
    cp "$file" "$DEST_DIR/${filename%.csv}_$counter.csv"
    # Increment the counter
    ((counter++))
done