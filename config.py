cfg = {
    "TensorFlowEasy": {
	"start_video": ["guvcview", "--device=/dev/video0", "-ao", "none", "-f", "YUYV", "-x", "320x308"],
	"set_mapping": "setmapping 36\n",
	"set_serout": "setpar serout USB\n" ,
	"set_serstyle": "D2",
	"set_threshold": "setpar thresh 35\n",
        "fpm": 30
    },
    "DarkNetYOLO": {
	"start_video": ["guvcview", "--device=/dev/video0", "-ao", "none", "-f", "YUYV", "-x", "448x240"],
	"set_mapping": "setmapping 14\n",
	"set_serout": "setpar serout USB\n" ,
	"set_serstyle": "D2",
	"set_threshold": "setpar thresh 15\n",
        "fpm": 3
    }
}
