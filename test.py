import os
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from utils import  get_values_from_batch


def get_preds(logits):
	class_output = F.softmax(logits, dim=1)
	pred_task = class_output.data.max(1, keepdim=True)[1]
	return pred_task

def test(dataloader, feature_extractor, task_classifier, disc_list, device, source_target, epoch = 0, tb_writer=None, Loss = F.binary_cross_entropy_with_logits):

	#print('\n\n testing \n')

	feature_extractor = feature_extractor.eval()
	task_classifier = task_classifier.eval()
	for disc in disc_list:
		disc = disc.eval()	
	
	with torch.no_grad():

		feature_extractor = feature_extractor.to(device)
		task_classifier = task_classifier.to(device)

		for disc in disc_list:
			disc = disc.to(device)

		target_iter = tqdm(enumerate(dataloader))

		n_total = 0
		n_correct = 0
		predictions_domain = []
		labels_domain = []

		for t, batch in target_iter:

			#if t > 100:
			#	break

			if source_target == 'source':
				#x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, y_domain_1, y_domain_2, y_domain_3 = batch
				#inputs = torch.cat((x_1, x_2, x_3), dim=0)
				#labels = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)
				inputs, labels, y_domain = get_values_from_batch(batch)
				#y_domain = torch.cat((y_domain_1, y_domain_2, y_domain_3), dim=0)
				y_domain.to(device)
			else:
				inputs, labels, y_domain = get_values_from_batch(batch)


			#print('labels.shape')
			#print(labels.shape)

			taille = inputs.size(2)

			inputs = inputs.to(device)
			labels = labels.to(device)


			
			features = feature_extractor.forward(inputs)
			
			# Task 
			task_out = task_classifier.forward(features)
			#print('task_out.shape')
			#print(task_out.shape)
			#print(task_out.shape)
			#print(task_out)
			#print('\n\n')
			task_out = F.upsample(task_out, taille, mode='linear')



			pred_task = get_preds(task_out)
			target_task = get_preds(labels)
			n_correct += pred_task.eq(target_task.data.view_as(pred_task)).cpu().sum() / target_task.size(2)
			#class_output = F.softmax(task_out, dim=1)
			#pred_task = class_output.data.max(1, keepdim=True)[1]
			#n_correct += pred_task.eq(y.data.view_as(pred_task)).cpu().sum()

			task_loss = Loss(task_out, labels)
			n_total += inputs.size(0)
			
			if source_target == 'source':
				# Domain classification
				for i, disc in enumerate(disc_list):
					pred_domain = disc.forward(features).squeeze()
					curr_y_domain = torch.where(y_domain == i, torch.ones(y_domain.size(0)), torch.zeros(y_domain.size(0))).float().to(device)
					try:
						predictions_domain[i] = torch.cat(predictions_domain[i], pred_domain)
						labels_domain[i] = torch.cat(labels_domain[i], curr_y_domain)
					except:
						predictions_domain.append(pred_domain)
						labels_domain.append(curr_y_domain)	
			try:
				predictions_task = torch.cat((predictions_task, pred_task), 0)
			except:
				predictions_task = pred_task

		acc = n_correct.item() * 1.0 / n_total			
		
		if tb_writer is not None:
			predictions_task_numpy = predictions_task.cpu().numpy()
			tb_writer.add_histogram('Test/'+source_target, predictions_task_numpy, epoch)
			tb_writer.add_scalar('Test/'+source_target+'_accuracy', acc, epoch)
			
			if source_target == 'source':
				for i, disc in enumerate(disc_list):
					predictions_domain_numpy = predictions_domain[i].cpu().numpy()
					labels_domain_numpy = labels_domain[i].cpu().numpy()
					#tb_writer.add_histogram('Test/source-D{}-pred'.format(i), predictions_domain[i], epoch)
					#tb_writer.add_pr_curve('Test/ROC-D{}'.format(i), labels = labels_domain_numpy, predictions = predictions_domain_numpy, global_step = epoch)
		return acc


def launch():

	return