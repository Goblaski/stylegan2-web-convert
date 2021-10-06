# Author G. de Jong (GITHUB: Goblaski)
# Parts of code used from https://github.com/NVlabs/stylegan2-ada-pytorch

import dnnlib
import numpy as np
import torch
import legacy
import functools
import click
import os

@click.command()
@click.option('--source', help='Input pkl file', required=True, metavar='PATH')
@click.option('--dest', help='Output folder', required=True, metavar='PATH')
def convert_to_onnx(source, dest):
    # load model on cpu
    device = torch.device('cpu')
    with dnnlib.util.open_url(source) as f:
        GG = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    # enforce 32 bit floats for full model (might be redundant in later version)
    GG.forward = functools.partial(GG.forward, force_fp32=True)
        
    #use a dummy input and label to determine graph 
    dummy_input = torch.from_numpy(np.random.RandomState(0).randn(1, GG.z_dim)).to(device)
    mapped_input = torch.from_numpy(np.random.RandomState(0).randn(1, 16, GG.z_dim)).to(device)
    label = torch.zeros([1, GG.c_dim], device=device)
    
    # export full model as onnx file
    in_names = [ "z" ] + [ "c"]
    out_names = [ "Y" ]
    torch.onnx.export(model=GG, 
                      args=(dummy_input,label),
                      f=os.path.join(dest,"model_full.onnx"),
                      input_names=in_names,
                      output_names=out_names,
                      verbose=True,
                      opset_version=10,
                      export_params=True,
                      do_constant_folding=False,
                      use_external_data_format=False,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
    
    # export mapping model as onnx file
    in_names_mapping = [ "z" ] + [ "c"]
    out_names_mapping = [ "mapped" ]
    
    torch.onnx.export(model=GG.mapping, 
                      args=(dummy_input, label),
                      f=os.path.join(dest,"model_mapping.onnx"),
                      input_names=in_names_mapping,
                      output_names=out_names_mapping,
                      verbose=True,
                      opset_version=10,
                      export_params=True,
                      do_constant_folding=False,
                      use_external_data_format=False,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
    
    # export synthesis model as onnx file
    GG.synthesis.forward = functools.partial(GG.synthesis.forward, force_fp32=True)
    in_names_synthesis = [ "mapping"]
    out_names_synthesis = [ "Y" ]
    
    torch.onnx.export(model=GG.synthesis, 
                      args=(mapped_input),
                      f=os.path.join(dest,"model_synthesis.onnx"),
                      input_names=in_names_synthesis,
                      output_names=out_names_synthesis,
                      verbose=True,
                      opset_version=10,
                      export_params=True,
                      do_constant_folding=False,
                      use_external_data_format=False,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    print("All done!")


if __name__ == "__main__":
    convert_to_onnx()