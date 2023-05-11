#!/usr/bin/env python3

import datarobot as dr
dr.Client(config_path = 'drconfig.yaml')

dr.Dataset.create_from_file(file_path="../hands_no_img.csv", categories=['TRAINING'])
