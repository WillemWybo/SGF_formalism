

def write_simple_axon(nseg, radius, node_len, internode_len):
	f = open('Moore1978_axon.swc', 'w')
	# soma
	f.write('1 1 0 0 0 12.5 -1 \n2 1 0 10 0 12.5 1 \n3 1 0 -10 0 12.5 1\n')
	# first node of ranvier
	x0 = node_len
	index = 4
	f.write('4 2 ' + str(x0) + ' 0 0 ' + str(radius) + ' 1\n')
	for i in range(nseg):
		x0 += internode_len
		index += 1
		f.write(str(index) + ' 2 ' + str(x0) + ' 0 0 ' + str(radius) + ' ' + str(index-1) + '\n')
		x0 += node_len
		index += 1
		f.write(str(index) + ' 2 ' + str(x0) + ' 0 0 ' + str(radius) + ' ' + str(index-1) + '\n')

	f.close()

if __name__ == '__main__':
	write_simple_axon(20, 5., 3.183, 2000.) # (Moore et al., 1978)
	# write_simple_axon(1, .5, 3.183, 2000.)