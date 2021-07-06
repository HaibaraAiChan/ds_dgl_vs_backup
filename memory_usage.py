import torch
import nvidia_smi



def see_memory_usage(message, force=True):
	logger = ''
	logger += message

	nvidia_smi.nvmlInit()
	handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
	info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + "GB\n"

	# logger += str(torch.cuda.memory_snapshot())
	# logger +=       '\nMemory Allocated '+str(torch.cuda.memory_allocated() / (1024 * 1024 * 1024)) +'  GigaBytes\n'
	logger += '\nMax Memory Allocated ' + str(
		torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)) + '  GigaBytes\n'
	# logger +=       'Cache Allocated '+str(torch.cuda.memory_cached() / (1024 * 1024 * 1024))+'  GigaBytes\n'
	# logger +=       'Max cache Allocated '+str(torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))+'  GigaBytes\n'
	print(logger)


# input("Press Any Key To Continue ..")