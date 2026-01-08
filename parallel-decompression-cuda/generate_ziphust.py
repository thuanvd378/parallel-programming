"""
ZIPHUST File Generator
Generates 10000 compressed .ziphust files for GPU vs CPU benchmark

Format:
  Header: "ZPHT" (4 bytes) + original_size (4) + compressed_size (4) + reserved (4)
  Data: Simple RLE encoding

Usage: python generate_ziphust.py [output_folder] [num_files]
"""

import os
import sys
import struct
import random
import time

MAGIC = b'ZPHT'

def rle_compress(data: bytes) -> bytes:
    """Simple RLE compression for benchmark purposes"""
    if len(data) == 0:
        return b''
    
    result = bytearray()
    i = 0
    
    while i < len(data):
        # Check for run of same bytes (at least 3)
        run_len = 1
        while i + run_len < len(data) and data[i + run_len] == data[i] and run_len < 64:
            run_len += 1
        
        if run_len >= 3:
            # Run encoding: 0x80 + (run_len - 3), followed by the byte
            result.append(0x80 + (run_len - 3))
            result.append(data[i])
            i += run_len
        else:
            # Literal: collect up to 127 non-repeating bytes
            lit_start = i
            lit_len = 0
            
            while i + lit_len < len(data) and lit_len < 127:
                # Check if next position starts a run
                if i + lit_len + 2 < len(data):
                    if data[i + lit_len] == data[i + lit_len + 1] == data[i + lit_len + 2]:
                        break
                lit_len += 1
            
            if lit_len > 0:
                # Literal encoding: length (0x00-0x7F), followed by bytes
                result.append(lit_len)
                result.extend(data[lit_start:lit_start + lit_len])
                i += lit_len
    
    return bytes(result)

def generate_compressible_data(size: int) -> bytes:
    """Generate data with some compressible patterns"""
    data = bytearray(size)
    i = 0
    
    while i < size:
        pattern_type = random.randint(0, 2)
        
        if pattern_type == 0:
            # Run of same byte (3-30 bytes)
            run_len = min(random.randint(3, 30), size - i)
            byte_val = random.randint(0, 255)
            for j in range(run_len):
                data[i + j] = byte_val
            i += run_len
        elif pattern_type == 1:
            # Random bytes (5-50 bytes)
            rand_len = min(random.randint(5, 50), size - i)
            for j in range(rand_len):
                data[i + j] = random.randint(0, 255)
            i += rand_len
        else:
            # Repeating pattern (text-like)
            patterns = [b'Hello ', b'World ', b'Test ', b'Data ', b'ZIPHUST ']
            pattern = random.choice(patterns)
            repeat = min(random.randint(1, 5), (size - i) // len(pattern))
            for _ in range(repeat):
                for b in pattern:
                    if i < size:
                        data[i] = b
                        i += 1
    
    return bytes(data)

def create_ziphust_file(filepath: str, original_size: int):
    """Create a single .ziphust file"""
    # Generate compressible data
    original_data = generate_compressible_data(original_size)
    
    # Compress
    compressed_data = rle_compress(original_data)
    
    # Write file
    with open(filepath, 'wb') as f:
        # Header: magic + original_size + compressed_size + reserved
        header = MAGIC + struct.pack('<III', len(original_data), len(compressed_data), 0)
        f.write(header)
        f.write(compressed_data)
    
    return len(original_data), len(compressed_data)

def main():
    output_folder = sys.argv[1] if len(sys.argv) > 1 else "test_ziphust_files"
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Generating {num_files} .ziphust files in '{output_folder}'...")
    
    total_original = 0
    total_compressed = 0
    start_time = time.time()
    
    for i in range(num_files):
        # Random file size between 1KB and 100KB
        file_size = random.randint(1024, 100 * 1024)
        filepath = os.path.join(output_folder, f"file_{i:05d}.ziphust")
        
        orig_size, comp_size = create_ziphust_file(filepath, file_size)
        total_original += orig_size
        total_compressed += comp_size
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Generated {i + 1}/{num_files} files ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
    
    print(f"\nDone!")
    print(f"  Files: {num_files}")
    print(f"  Total original: {total_original / 1024 / 1024:.1f} MB")
    print(f"  Total compressed: {total_compressed / 1024 / 1024:.1f} MB")
    print(f"  Compression ratio: {ratio:.1f}%")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\nFolder: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    main()
