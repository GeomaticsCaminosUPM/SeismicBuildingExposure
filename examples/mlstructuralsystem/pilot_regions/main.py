import sys

import seismicbuildingexposure.mlstructuralsystem.dataset as dataset 
import seismicbuildingexposure.mlstructuralsystem.train as train 

import config

if __name__ == "__main__":
    try:
        dataset.preprocess(config)
        train.run(config)
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)