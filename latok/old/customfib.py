def fibonacci_like_sequence(start1, start2, n):
    """Generate a Fibonacci-like sequence starting with two given numbers."""
    sequence = [start1, start2]
    for _ in range(2, n):
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    return sequence

# Get user input
start1 = int(input("Enter the first number: "))
start2 = int(input("Enter the second number: "))
n = int(input("How many numbers in the sequence do you want? "))

# Generate and print the sequence
result = fibonacci_like_sequence(start1, start2, n)
print(f"\nFibonacci-like sequence starting with {start1} and {start2}:")
print(", ".join(map(str, result)))
