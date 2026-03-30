# reading formal data sets
base_path = '/kaggle/input/ai-vs-real-images-dataset'

# Define paths for AI-generated and Real images
ai_path = os.path.join(base_path, 'Ai_generated_dataset')
real_path = os.path.join(base_path, 'real_dataset')


ai = os.path.join(ai_path , 'people')
ai_people_path = os.path.join(ai_path, 'people')
real_people_path = os.path.join(real_path, 'people')

# Verify paths exist
print(f"AI People path exists: {os.path.exists(ai_people_path)}")
print(f"Real People path exists: {os.path.exists(real_people_path)}")
print("fprmal aveliabel data ")
if os.path.exists(ai_path):
    print([d for d in os.listdir(ai_path) if os.path.isdir(os.path.join(ai_path, d))])

print("\nAvailable Real categories:")
if os.path.exists(real_path):
    print([d for d in os.listdir(real_path) if os.path.isdir(os.path.join(real_path, d))])