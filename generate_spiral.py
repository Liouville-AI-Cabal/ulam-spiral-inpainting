import numpy as np
import matplotlib.pyplot as plt
import time
import math

def generate_primality_values_numpy(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    print("Completed prime count.")
    return is_prime

def draw_fast_spiral_imshow(values, image_size):
    image_data = np.zeros((image_size, image_size), dtype=np.uint8)
    
    center = image_size // 2
    x, y = 0, 0
    dx, dy = 1, 0
    segment_length, segment_passed, turn_counter = 1, 0, 0

    values[1] = True

    for i in range(image_size * image_size):
        number_to_plot = i + 1
        
        if values[number_to_plot]:
            row, col = center - y, center + x
            if 0 <= row < image_size and 0 <= col < image_size:
                image_data[row, col] = 255
        
        x, y = x + dx, y + dy
        segment_passed += 1
        
        if segment_passed == segment_length:
            dx, dy = -dy, dx
            segment_passed = 0
            turn_counter += 1
            if turn_counter % 2 == 0:
                segment_length += 1

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.imshow(image_data, cmap='gray', interpolation='none')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig

if __name__ == "__main__":
    IMAGE_SIZE = 14143
    
    TOTAL_NUMBERS = IMAGE_SIZE * IMAGE_SIZE

    print(f"Generate Ulam Spiral image of size {IMAGE_SIZE}x{IMAGE_SIZE}...")

    start_time = time.time()
    
    prime_values = generate_primality_values_numpy(TOTAL_NUMBERS)
    
    figure = draw_fast_spiral_imshow(prime_values, IMAGE_SIZE)
    
    dpi = 100
    figure.set_size_inches(IMAGE_SIZE / dpi, IMAGE_SIZE / dpi)

    file_name = f"ulam_spiral_optimal_{IMAGE_SIZE}x{IMAGE_SIZE}.png"
    print(f"Saving image as '{file_name}' with resolution {IMAGE_SIZE}x{IMAGE_SIZE} pixels...")

    figure.savefig(file_name, dpi=dpi, facecolor='black')
    
    end_time = time.time()

    print(f"Completed! Image saved as '{file_name}'")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

    plt.close(figure)
