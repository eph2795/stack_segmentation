data_conf = {
    'conf_name': 'soil_stacks',
    'stacks': [
#         {
#             'path': '../../soil_data/Biomat3',
#             'slice_train': (slice(None), slice(None), slice(455)),
#             'slice_val': (slice(None), slice(None), slice(480, None)),
#         },
        {
            'path': '../../soil_data/CF11_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/CF12_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/CF14_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/CF3_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/CF9_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/REV_sample1_1_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        },
        {
            'path': '../../soil_data/REV_sample2_1_700',
            'slice_train': (slice(None), slice(None), slice(455)),
            'slice_val': (slice(None), slice(None), slice(480, None)),
        }
    ],
    'patches': {
        'train': (128, 128, 1),
        'val': (128, 128, 1),
        'test': (128, 128, 1),
    },
}