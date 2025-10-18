# Define the weights and charges for each variable
weights = {'a': 678.69, 'b': 636.73, 'c': 65.48, 'd': 149.07}
charges = {'a': 0, 'b': 0, 'c': 2, 'd': -1}

# Open the file to write the combinations
with open('combination.txt', 'w') as file:
    # Iterate through all possible combinations of a, b, c, and d (0 to 5)
    for a in range(6):
        for b in range(6):
            for c in range(6):
                for d in range(6):
                    # Calculate the total weight and charge for the current combination
                    total_weight = a * weights['a'] + b * weights['b'] + c * weights['c'] + d * weights['d']
                    total_charge = a * charges['a'] + b * charges['b'] + c * charges['c'] + d * charges['d']
                    
                    # Write the combination and its total weight and charge to the file
                    file.write(f"{a}a + {b}b + {c}c + {d}d = weight {total_weight:.2f}, charge {total_charge}, weight/charge {total_weight/ (total_charge + 0.0001)}\n")