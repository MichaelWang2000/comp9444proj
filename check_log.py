import torch
from tensorboard.backend.event_processing import event_accumulator

# 读取事件文件路径
event_file_path = 'events.out.tfevents.1721398922.7ed9d07b60e5'

# 创建一个事件累加器
event_acc = event_accumulator.EventAccumulator(event_file_path)
event_acc.Reload()

# 获取所有标量标签
tags = event_acc.Tags()['scalars']

# 打印所有标量标签
print("Tags:", tags)

# 打印每个标量标签对应的事件数据
for tag in tags:
    events = event_acc.Scalars(tag)
    for event in events:
        print(f"Tag: {tag}, Step: {event.step}, Value: {event.value}")