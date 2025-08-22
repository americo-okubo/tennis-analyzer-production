from tennis_comparison_backend import TennisComparisonEngine
import shutil
import tempfile
import os

analyzer = TennisComparisonEngine()

# Test like the API does - copy file to temp location
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
    # Copy our test file to temp location
    shutil.copy2('videos/Americo_FD_D_E.mp4', temp_file.name)
    temp_path = temp_file.name

print('Testing with temp file:', temp_path)

# Test the extraction method with temp file (like API does)
user_file_movement = analyzer._extract_movement_from_professional(temp_path, {})
print('File movement from temp:', user_file_movement)

user_config_movement = 'backhand_drive'  # What the API would build
print('Config movement:', user_config_movement)

print('Match:', user_file_movement == user_config_movement)
print('Not unknown:', user_file_movement != 'unknown')
should_reject = user_file_movement != 'unknown' and user_file_movement != user_config_movement
print('Should reject:', should_reject)

# Cleanup
os.unlink(temp_path)