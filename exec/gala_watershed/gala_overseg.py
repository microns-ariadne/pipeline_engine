import sys
import libtiff
from gala_ripped import segmentation_pipeline

if __name__ == "__main__":
    sys.exit(segmentation_pipeline.entrypoint(sys.argv))
