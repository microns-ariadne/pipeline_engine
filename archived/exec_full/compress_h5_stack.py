
import sys
import os
import h5py

def execute(input_h5_filename, output_h5_filename):
	
	print 'reading = %s' % (input_h5_filename,)
	
	f_in = h5py.File(input_h5_filename, 'r')
	dset = f_in['stack']
	
	print 'dset.shape = %r' % (dset.shape,)
	
	print 'writing = %s' % (output_h5_filename,)
	f_out = h5py.File(output_h5_filename, 'w')
	f_out.create_dataset('stack', data=dset, compression="gzip")
	


if '__main__' == __name__:
	try:
		prog_name, input_h5_filename, output_h5_filename = sys.argv[:3]
		
	except ValueError, e:
		sys.exit('USAGE: %s [input_h5_filename] [output_h5_filename] ' % (sys.argv[0],))

	execute(input_h5_filename, output_h5_filename)

