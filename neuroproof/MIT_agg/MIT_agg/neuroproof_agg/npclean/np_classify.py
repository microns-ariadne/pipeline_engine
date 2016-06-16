import sys
import subprocess
import time
from optparse import OptionParser

# usage: python np_classify.py <options> <EM image stack> <prob map> <groundtruth>
#        <groundtruth> is only needed of option --do-compare is True
# Note: if not generating watershed, need to place 'supervoxels.h5' in same directory

parser = OptionParser()
parser.add_option("--gen-boundpred", default='True', help="Choose to generate or provide boundary prediction. Default is True.", metavar=True)
parser.add_option("--do-compare", default='False', help="Do compare to groundtruth (provide as last argument). Default is False.", metavar=False)
parser.add_option("--do-test", default='False', help="Do test against the original binary. Default is False.", metavar=False)
parser.add_option("--do-ws", default='True', help="Generate supervoxels.h5. Default is True", metavar=True)

(options, args) = parser.parse_args()

# GALA PART

# image_stack = "'/root/NeuroProof/examples/training_sample2/grayscale_maps/*.png'"
# image_stack = "'/root/NeuroProof/examples/validation_sample/grayscale_maps/*.png'"

# image_stack = "/scratch/neuro_segmentation/ISBI_data/train-input.tif"
# image_stack = "/scratch/neuro_segmentation/ISBI_data/test-input.tif"
# image_stack = "/scratch/neuro_segmentation/ISBI_data/test-input-coarse.tif"
# image_stack = "/scratch/neuro_segmentation/ISBI_data/test-input-512-bs.tif"
# ilp_file = "/root/NeuroProof/examples/training_sample1/results/boundary_classifier_ilastik.ilp"

image_stack = args[0]

boundary_prediction = ""

if options.do_ws == 'True':

	if options.gen_boundpred == 'True':
		cmd = "gala-segmentation-pipeline -I " + image_stack + " --ilp-file " + ilp_file + " --enable-gen-supervoxels --enable-gen-pixel --enable-h5-output --seed-size 5 . --segmentation-thresholds 0.0" 
		boundary_prediction = "/scratch/neuro_segmentation/STACKED_prediction.h5"
	else:
		if len(args) < 2:
			print "ERROR: Please provide pixelprob_file"
			exit(0)
		boundary_prediction = args[1]
		cmd = "gala-segmentation-pipeline -I " + image_stack + " --enable-gen-supervoxels --disable-gen-pixel --pixelprob-file=" + boundary_prediction + " --enable-h5-output --seed-size 5 . --segmentation-thresholds 0.0" 

	start_gen_oversegmentation = time.time()
	print cmd
	print subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
	end_gen_oversegmentation = time.time()
else:
    boundary_prediction = args[1]


# print "Oversegmentation generation:", end_gen_oversegmentation - start_gen_oversegmentation
# exit(0)


# NEUROPROOF PART

# use result from GALA:
oversegmented_labels = "/home/vj/neuroproof/data/isbi/supervoxels.h5"
# boundary_prediction = "/root/STACKED_prediction.h5"
# boundary_prediction = "/scratch/neuro_segmentation/ISBI_data/test-membranes-idsia-coarse.h5"


# oversegmented_labels = "/root/NeuroProof/examples/training_sample2/oversegmented_stack_labels.h5"
# oversegmented_labels = "gala_demo/demo-train-ws.h5"
# oversegmented_labels = "examples/validation_sample/oversegmented_stack_labels.h5"
# boundary_prediction = "/root/NeuroProof/examples/training_sample2/boundary_prediction.h5"
# boundary_prediction = "gala_demo/demo-train-pr.h5"
# groundtruth = "/root/NeuroProof/examples/validation_sample/groundtruth.h5"
# groundtruth = "/root/NeuroProof/examples/training_sample2/groundtruth.h5"
# groundtruth = "gala_demo/demo-train-gt.h5"

# groundtruth_validation = groundtruth

print "==> Agglomeration procedure ..."
classifier = "classifier.xml"
cmd = "/home/vj/npclean/build/neuroproof_graph_predict --agglo-type 5 --num-top-edges 256"
cmd_line = [cmd, oversegmented_labels, boundary_prediction, classifier]
print "starting: " + str(cmd_line)

start_agglomeration = time.time()
print subprocess.Popen(' '.join(cmd_line), shell=True, stdout=subprocess.PIPE).stdout.read()
end_agglomeration = time.time()


# print "==> Graph analysis ..."
# cmd = "/root/NeuroProof/build/bin/neuroproof_graph_analyze"
# cmd_line = [cmd, "--graph-file graph.json -g 1 -b 1"]

# print subprocess.Popen(' '.join(cmd_line), shell=True, stdout=subprocess.PIPE).stdout.read()

if options.do_compare == 'True':
    groundtruth_validation = args[2]
    print "==> Compare to groundtruth ..."
    segmentation = "segmentation.h5"
    cmd = "sudo /scratch/neuro_segmentation/NeuroProof/build/bin/neuroproof_graph_analyze_gt"
    cmd_line = [cmd, segmentation, groundtruth_validation]

    print subprocess.Popen(' '.join(cmd_line), shell=True, stdout=subprocess.PIPE).stdout.read()

if options.do_test == 'True':
    print "==> Running test against original binary ..."
    subprocess.Popen("mv segmentation.h5 segmentation.h5.to_be_compared", shell=True, stdout=subprocess.PIPE).communicate()
    cmd = "sudo ./test_suite/neuroproof_graph_predict --agglo-type 5"
    cmd_line = [cmd, oversegmented_labels, boundary_prediction, classifier]
    subprocess.Popen(' '.join(cmd_line), shell=True, stdout=subprocess.PIPE).communicate()

    # test 1
    out, err = subprocess.Popen("diff segmentation.h5 segmentation.h5.to_be_compared", shell=True, stdout=subprocess.PIPE).communicate()
    if err:
        print "TEST 1: Failed. Error occurred."
    else:
        if out == "":
            print "TEST 1: Passed. Segmented volume is same as validation volume."
        else:
            print "TEST 1: Failed. Segmented volume differs from validation volume."

    subprocess.Popen("mv segmentation.h5 segmentation.h5.test", shell=True, stdout=subprocess.PIPE).communicate()
    subprocess.Popen("mv segmentation.h5.to_be_compared segmentation.h5", shell=True, stdout=subprocess.PIPE).communicate()
    


# CLEAN UP
print "Cleaning up .........."
# subprocess.Popen('rm config.json', shell=True, stdout=subprocess.PIPE)
# subprocess.Popen('rm graph.json', shell=True, stdout=subprocess.PIPE)
# subprocess.Popen('rm STACKED_prediction.h5', shell=True, stdout=subprocess.PIPE)

# print "Oversegmentation generation:", end_gen_oversegmentation - start_gen_oversegmentation
print "Agglomeration procedure:    ", end_agglomeration - start_agglomeration

