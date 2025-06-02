from ase.io import read, write

datafile = 'datasets/test.xyz'
datafile_pached= 'datasets/test_pached.xyz'
for struct in read(datafile, format='extxyz', index=':'):
    if 'config_type' in struct.info:
            config_type = struct.info['config_type']
            if 'liquid' in config_type:
                struct.info['config_type'] = 'liquid'
                write(datafile_pached, struct, append=True)
            else: 
                write(datafile_pached, struct, append=True)