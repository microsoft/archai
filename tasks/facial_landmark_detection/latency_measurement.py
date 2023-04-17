import io
from typing import Dict, List, Optional, Tuple, Union

import os
os.environ["OMP_NUM_THREADS"] = "1"
import onnxruntime as rt
import psutil
import torch

from archai.common.timing import MeasureBlockTime
from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives

import os
import statistics
from math import sqrt
from multiprocessing import Process, Queue
from time import perf_counter, sleep
from typing import Dict, Optional, Tuple


class AvgOnnxLatency:
    higher_is_better: bool = False

    def __init__(self, input_shape: Union[Tuple, List[Tuple]], num_trials: int = 1, num_input: int = 10,
                 input_dtype: str = 'torch.FloatTensor', rand_range: Tuple[float, float] = (0.0, 1.0),
                 export_kwargs: Optional[Dict] = None, inf_session_kwargs: Optional[Dict] = None):
        """Uses the average ONNX Latency (in seconds) of an architecture as an objective function for
        minimization.

        Args:
            input_shape (Union[Tuple, List[Tuple]]): Model Input shape or list of model input shapes.
            num_trials (int, optional): Number of trials. Defaults to 1.
            export_kwargs (Optional[Dict], optional): Optional dictionary of key-value args passed to
                `torch.onnx.export`. Defaults to None.
            inf_session_kwargs (Optional[Dict], optional): Optional dictionary of key-value args 
                passed to `onnxruntime.InferenceSession()`. Defaults to None.
        """
        input_shapes = [input_shape] if isinstance(input_shape, tuple) else input_shape            
        
        rand_min, rand_max = rand_range
        self.sample_input = tuple([
            ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type(input_dtype)
            for input_shape in input_shapes
        ])

        self.input_dtype = input_dtype
        self.rand_range = rand_range
        self.num_trials = num_trials
        self.num_input_per_trial = num_input
        self.export_kwargs = export_kwargs or dict()
        self.inf_session_kwargs = inf_session_kwargs or dict()

    def evaluate(self, model: ArchaiModel, dataset_provider: DatasetProvider = None,
                budget: Optional[float] = None) -> float:
        model.arch.to('cpu')

        # Exports model to ONNX
        exported_model_buffer = io.BytesIO()
        torch.onnx.export(
            model.arch, self.sample_input, exported_model_buffer,
            input_names=[f'input_{i}' for i in range(len(self.sample_input))],
            opset_version=11,
            **self.export_kwargs
        )
        print("torch.onnx.export done")
        exported_model_buffer.seek(0)

        opts = rt.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # Benchmarks ONNX model
        onnx_session = rt.InferenceSession(exported_model_buffer.read(), sess_options=opts, **self.inf_session_kwargs)
        sample_input = {f'input_{i}': inp.numpy() for i, inp in enumerate(self.sample_input)}
        # inf_times = []

        # for _ in range(self.num_trials):
        #     with MeasureBlockTime('onnx_inference') as t:
        #         onnx_session.run(None, input_feed=sample_input)
        #     inf_times.append(t.elapsed)

        # return sum(inf_times) / self.num_trials

        num_input_per_trial = self.num_input_per_trial
        #inf_time_avg, inf_time_std = self.get_model_latency_1cpu(onnx_session, model.arch, sample_input, cpu = 1, onnx = True, 
        inf_time_avg, inf_time_std = self.get_time_elapsed (onnx_session, model.arch, sample_input, onnx = True, num_input = num_input_per_trial, num_measures = self.num_trials)

        #per trial time is idealy longer so that timing can be more accurate
        if (inf_time_avg * num_input_per_trial < 100) : 
            num_input_per_trial = int(1.5 * 100 / inf_time_avg + 0.5)
            #inf_time_avg, inf_time_std = self.get_model_latency_1cpu(onnx_session, model.arch, sample_input, cpu = 1, onnx = True, num_input = num_input_per_trial, num_measures = self.num_trials)
            inf_time_avg, inf_time_std = self.get_time_elapsed (onnx_session, model.arch, sample_input, onnx = True, num_input = num_input_per_trial, num_measures = self.num_trials)

        if (inf_time_std > 0.1 * inf_time_avg):
            ratio = (0.1 * inf_time_avg) / inf_time_std
            ratio *= ratio * 1.1 
            num_trails_scaled = int(self.num_trials * ratio + 0.5)
            #inf_time_avg, inf_time_std = self.get_model_latency_1cpu(onnx_session, model.arch, sample_input, cpu = 1, onnx = True, num_input = num_input_per_trial, num_measures = num_trails_scaled)
            inf_time_avg, inf_time_std = self.get_time_elapsed (onnx_session, model.arch, sample_input, onnx = True, num_input = num_input_per_trial, num_measures = self.num_trials)

        assert (inf_time_std < 0.1 * inf_time_avg,  f"inf_time_std = {inf_time_std}, inf_time_avg = {inf_time_avg:}")
        return inf_time_avg

    def get_model_latency_1cpu(self, onnx_session, model, sample_input, cpu: int, onnx: bool = False, num_input:int = 15, num_measures:int = 15) -> Tuple[float, float] :
        """ runs the timing measurement code on 1 cpu (use cpu #1 since that is what I have been using"""
        # IMPORTANT: the next two lines as a hack are needed for many multiproc issues torch has 
        torch.set_num_threads(1) #do this even though it maybe already be 1
        assert (torch.get_num_threads() == 1)
        sleep(1.0) #not sure why this is needed but it *is*, otherwise some onnx_session.run() could never return from the capi call

        return_q = Queue()
        p = Process(target=self.get_time_elapsed_process, args=(return_q, cpu, onnx_session, model, sample_input, onnx, num_input, num_measures))
        p.start()
        p.join()
        ret = return_q.get()
        return ret

    def get_time_elapsed_process (self, return_q, cpu, onnx_session, model, sample_input, onnx: bool = False, num_input:int = 15, num_measures:int = 15):
        """ wrapper to handle return values"""

        assert (cpu < os.cpu_count())
        # pin created processes 
        pid = os.getpid()
        assert (pid != None)
        print(f"get_time_elapsed_process pid: {pid}")
        
        p = psutil.Process(pid)
        print(f"Setting CPU affinity for pid {pid} to CPU {cpu}")
        p.cpu_affinity([cpu])
    
        ret = self.get_time_elapsed (onnx_session, model, sample_input, onnx, num_input, num_measures)
        return_q.put (ret)

    def get_time_elapsed (self, onnx_session, model, sample_input, onnx: bool = False, num_input:int = 15, num_measures:int = 15) -> Tuple[float, float] :
        #print("get_time_elapsed: entering")
        def meausre_func() : 
            #print(f"measure_func entered")
            t0 = perf_counter()
            for _ in range(num_input): #this is to incease the accuracy as 1 run maybe too short to measure
                #print(f"measure_func iter: {_}")
                if (onnx):
                    #print(f"measure_func: start onnx_session.run")
                    onnx_session.run(None, input_feed=sample_input)[0]
                    #print(f"measure_func: left onnx_session.run")
                else:
                    pred = model.forward(sample_input)
            t1 = perf_counter()                
            time_measured = 1e3 * (t1 - t0) / num_input
            #print(f"measure_func return: {time_measured}")
            return time_measured


        time_measured_all = []
        #print("get_time_elapsed: starting measure_func")
        time_measured_all = [meausre_func() for _ in range (num_measures)]
        time_measured_avg = statistics.mean(time_measured_all)
        time_measured_std = sqrt(num_input) * statistics.stdev(time_measured_all) #sigma^2(x+y) = sigma^2(x) + sigma^2(y); then there is average
        time_measured_std /= sqrt(num_measures) #stdev of the sample mean, not the population
        #print(f"get_time_elapsed return: {time_measured_avg}, {time_measured_std}")
        return time_measured_avg, time_measured_std #, time_measured_all