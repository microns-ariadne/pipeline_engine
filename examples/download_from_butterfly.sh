#! /bin/bash
#
# download_from_butterfly.sh
#
# This example downloads a volume of data from Butterfly into a dataset
# named "images" in the directory, $MICRONS_DIR. The example uses a local
# scheduler and you do not need to run the luigi daemon or Butterfly.
#
# Environment variables needed:
#
# MICRONS_DIR - the destination for the volume
#

#
# Environment variables for the script.
#
# The URL for the butterfly server's API
#
if [ -z "$BUTTERFLY_API_URL" ]; then
   BUTTERFLY_API_URL=https://butterfly.rc.fas.harvard.edu/api
fi
#
# The coordinates to fetch
#
MICRONS_X=0
MICRONS_Y=0
MICRONS_Z=0
MICRONS_WIDTH=1496
MICRONS_HEIGHT=1496
MICRONS_DEPTH=50
#
# The details for connecting to the dataset on butterfly
#
MICRONS_EXPERIMENT=ECS_train_images
MICRONS_SAMPLE=neocortex
MICRONS_DATASET=sem
MICRONS_CHANNEL=raw
#
# The details for the dataset
#
MICRONS_DATASET_NAME=image
#
# The command
#
luigi --module ariadne_microns_pipeline.tasks \
      --local-scheduler \
      ariadne_microns_pipeline.DownloadFromButterflyTask \
      --volume='{"x":'$MICRONS_X',"y":'$MICRONS_Y',"z":'$MICRONS_Z',"width":'$MICRONS_WIDTH',"height":'$MICRONS_HEIGHT',"depth":'$MICRONS_DEPTH'}' \
      --destination='{"roots":["'$MICRONS_DIR'"],"dataset_name":"'$MICRONS_DATASET_NAME'","pattern":"{x:09d}_{y:09d}_{z:09d}_image"}' \
      --experiment=$MICRONS_EXPERIMENT \
      --sample=$MICRONS_SAMPLE \
      --dataset=$MICRONS_DATASET \
      --channel=$MICRONS_CHANNEL \
      --url=$BUTTERFLY_API_URL
			     
