class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
        
    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class OpenDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df_path, height, width, img_transforms=None):
        self.transforms = img_transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path)
#         self.df = self.df[:1000]
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)
        
        # Filling up image_info is left as an exercise to the reader
        
        counter = 0
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row["ImageID"]
            image_path = os.path.join(self.image_dir, image_id)

            if os.path.exists(image_path + '.jpg'):
                self.image_info[counter]["image_id"] = image_id
                self.image_info[counter]["image_path"] = image_path
                self.image_info[counter]["XMin"] = row["XMin"]
                self.image_info[counter]["YMin"] = row["YMin"]
                self.image_info[counter]["XMax"] = row["XMax"]
                self.image_info[counter]["YMax"] = row["YMax"]
                self.image_info[counter]["labels"] = row["LabelEncoded"]
                counter += 1

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"] + ".jpg"
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        # processing part and extraction of boxes is left as an exercise to the reader
        # get bounding box coordinates for each mask         
        num_objs = len(self.image_info[idx]["labels"].split())
        
        boxes = []
        xmins = self.image_info[idx]["XMin"].split()
        ymins = self.image_info[idx]["YMin"].split()
        xmaxs = self.image_info[idx]["XMax"].split()
        ymaxs = self.image_info[idx]["YMax"].split()
        
        assert len(xmins) == len(ymins) == len(xmaxs) == len(ymaxs) == num_objs
        
        for i in range(num_objs):
            xmin = float(xmins[i])
            xmax = float(xmaxs[i])
            ymin = float(ymins[i])
            ymax = float(ymaxs[i])
            boxes.append([xmin, ymin, xmax, ymax])
                                                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([int(x) for x in self.image_info[idx]["labels"].split()])

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                                                              
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = transforms.ToTensor()(img)

#         print(img)
#         print(target)
        return img, target

    def __len__(self):
        return len(self.image_info)

# def inference(model,data):
#     return

# def mAP():
#     return