import argparse
import numpy as np
import random
import time

if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='Hidden Polarized Communities - Noise Adder')

    # create and read the arguments
    parser.add_argument('d', help='dataset', type=str)
    parser.add_argument('n', help='noise', type=float)
    args = parser.parse_args()

    # get nodes, number of edges, and percentage of negative edges
    original_nodes = set()
    number_of_edges = 0
    number_of_negative_edges = 0

    with open('datasets/' + args.d + '.txt') as dataset_file:
        for line in dataset_file:
            if '#' in line:
                graph_size = int(line.strip('\n').split(' ')[1])
                continue

            split_line = line.split('\t')

            from_node = int(split_line[0])
            to_node = int(split_line[1])
            sign = int(split_line[2])

            original_nodes.add(from_node)
            original_nodes.add(to_node)

            number_of_edges += 1
            if sign == -1:
                number_of_negative_edges += 1

    print('Graph size: {}'.format(graph_size))
    new_graph_size = int(graph_size * (1+args.n))
    print('New graph size: {}'.format(new_graph_size))


    ##################################################
    # Old stuff. Can be simplified.
    # identify  the random nodes
    max_node_id = max(original_nodes)
    random_nodes = set(range(max_node_id + 1, max_node_id + int(len(original_nodes) * args.n) + 1))

    # get the average degree (rounded) and the ratio of negative edges
    avg_degree = int(round(float(number_of_edges) * 2 / len(original_nodes)))
    ratio_of_negative_edges = float(number_of_negative_edges) / number_of_edges

    print('avg d: {}'.format(avg_degree))
    print('neg ratio: {}'.format(ratio_of_negative_edges))

    # output file
    noised_dataset_file = open('datasets/' + args.d + '_noised_' + str(int(args.n)) + '.txt', 'w')

    # write the number of nodes
    noised_dataset_file.write('# ' + str(len(original_nodes) + len(random_nodes)) + '\n')

    # copy the original edges
    with open('datasets/' + args.d + '.txt') as dataset_file:
        for line in dataset_file:
            if '#' in line:
                continue
            noised_dataset_file.write(line)

    ##################################################


    # Sample!
    s = time.time()
    universe = (int(graph_size*(1+args.n)-1) * graph_size*(args.n))
    left = int(graph_size*(args.n))*avg_degree # Random edges left to generate
    sample = np.array([])
    done = False
    print('Sampling...')
    while not done:
        new_sample = np.random.randint(0, universe, left)
        sample = np.hstack([sample, new_sample])
        diff = len(sample) - len(np.unique(sample)) # Count repeats to resample
        done = diff == 0
        sample = np.unique(sample) # Remove repeats
        print('Remaining: {}'.format(diff))
        left = diff
    print('Sorting...')
    sample = np.sort(sample) % (graph_size*(1+args.n)-1)
    print('Flipping negatives...')
    negatives = np.random.choice(range(len(sample)), int(ratio_of_negative_edges*len(sample)), replace=False)
    sample[negatives] *= -1
    sample = sample.reshape((int(graph_size*(args.n)), avg_degree)).astype(np.int)
    print('Done')

    print('Time: {}'.format(time.time()-s))
    print('Sample size: {}'.format(sample.shape))
    counts = np.unique(np.abs(sample), return_counts=True)[1]
    print('Mean (std.) occurences: {} ({})'.format(np.mean(counts), np.std(counts)))
    print('min-nz, max occurences: {}, {}'.format(np.min(counts), np.max(counts)))
    print()

    print('Sanity check:')
    D = np.diff(np.abs(sample), axis=1)
    print('Duplicate edges? {}'.format(0 in D))
    print('min,max: {},{}'.format(np.min(sample), np.max(sample)))
    print('negatives ratio: {}'.format(np.sum(sample<0)/np.size(sample) ))
    print()

    print('Dumping...')
    # create and write the random edges
    possible_neighbors = (np.arange(new_graph_size)+1).astype(np.int)
    current_index = graph_size
    for random_node in random_nodes:
        current_row = sample[current_index-graph_size,:]
        neighbors = possible_neighbors[np.abs(current_row)]
        for j in range(len(neighbors)):
            sign = np.sign(current_row[j])
            noised_dataset_file.write(str(current_index) + '\t' + str(np.abs(neighbors[j])) + '\t' + str(sign) + '\n')

        possible_neighbors[current_index] -= 1
        current_index += 1

    # close the output file
    noised_dataset_file.close()

    print('Done')
