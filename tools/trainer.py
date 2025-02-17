import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tools.utils import *


class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, model_name, log_dir, num_views=6):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):
        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):

            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch

            for i, data in enumerate(self.train_loader):
                if self.model_name == 'OVPT':
                    # N batch_size V view C channels H height W weight
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)

                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 100 == 0:
                    print(log_str)
                i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc, cm = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                torch.save(self.model.state_dict(), self.log_dir + "/" + "model_%d.pth" % (epoch + 1))
                # self.model.save(self.log_dir, epoch+1)
                list_diag = np.diag(cm)
                list_raw_sum = np.sum(cm, axis=1)
                val_each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
                # print('val each acc ：', val_each_acc)
                plot_confusion_matrix(cm)
                record_output(val_overall_acc, val_mean_class_acc, val_each_acc, 'records.txt')

            # adjust learning rate manually and (epoch + 1) % M10 == 0
            if epoch > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

        # export scalar entropy to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # M40
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)

        # M10
        # wrong_class = np.zeros(10)
        # samples_class = np.zeros(10)

        all_loss = 0

        self.model.eval()

        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'OVPT':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            target = target.cpu().data.numpy()
            pred = pred.cpu().data.numpy()
            for j in range(len(pred)):
                all_target.append(target[j])
                all_pred.append(pred[j])

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)

        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        cm = confusion_matrix(all_target, all_pred)

        print('val overall acc. : ', val_overall_acc)
        print('val mean class acc. : ', val_mean_class_acc)
        print('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc, cm
