import numpy as np
import h5py
import os

class DataLoader:
    """Class minibatches from data on disk in HDF5 format"""
    def __init__(self, args, region_dim, phrase_dim, plh, split):
        """Constructor

        Arguments:
        args -- command line arguments passed into the main function
        region_dim -- dimensions of the region features
        phrase_dim -- dimensions of the phrase features
        plh -- placeholder dictory containing the tensor inputs
        split -- the data split (i.e. 'train', 'test', 'val')
        """
        datafn = os.path.join('data', args.dataset, '%s_imfeats.h5' % split)
        self.data = h5py.File(datafn, 'r')
        vecs = np.array(self.data['phrase_features'], np.float32)
        phrases = list(self.data['phrases'])
        assert(vecs.shape[0] == len(phrases))

        w2v_dict =  {}
        for index, phrase in enumerate(phrases):
            w2v_dict[phrase] =  vecs[index, :]

        # mapping from uniquePhrase to w2v
        self.w2v_dict = w2v_dict
        self.pairs = list(self.data['pairs'])
        self.n_pairs = len(self.pairs[0])
        self.phrases = phrases

        self.im2pairs = {}
        self.max_phrases = 0
        for sample_id in range(self.n_pairs):
            im_id = self.pairs[0][sample_id]
            if im_id not in self.im2pairs:
                self.im2pairs[im_id] = []

            self.im2pairs[im_id].append(sample_id)
            self.max_phrases = max(self.max_phrases, len(self.im2pairs[im_id]))

        self.im_ids = self.im2pairs.keys()
        self.n_pairs = len(self.im_ids)
        self.pair_index = list(range(self.n_pairs))

        self.split = split
        self.plh = plh
        self.is_train = split == 'train'
        self.neg_to_pos_ratio = args.neg_to_pos_ratio
        self.batch_size = args.batch_size
        self.max_boxes = args.max_boxes
        if self.is_train:
            self.success_thresh = args.train_success_thresh
        else:
            self.success_thresh = args.test_success_thresh

        self.region_feature_dim = region_dim
        self.phrase_feature_dim = phrase_dim

    def __len__(self):
        return self.n_pairs

    def shuffle(self):
        ''' Shuffles the order of the pairs being sampled
        '''
        np.random.shuffle(self.pair_index)

    def num_batches(self):
        return int(np.ceil(float(len(self)) / self.batch_size))

    def get_batch(self, batch_id):
        """Returns a minibatch given a valid id for it

        Arguments:
        batch_id -- number between 0 and self.num_batches()

        Returns:
        feed_dict -- dictionary containing minibatch data
        gt_labels -- indicates positive/negative regions
        num_pairs -- number of pairs without padding
        """
        region_features = np.zeros((self.batch_size, self.max_boxes,
                                    self.region_feature_dim), dtype=np.float32)
        num_pairs = self.batch_size
        start_pair = batch_id * num_pairs
        end_pair = min(start_pair + num_pairs, len(self))
        num_pairs = end_pair - start_pair

        im_ids = [self.im_ids[self.pair_index[start_pair + pair_id]] for pair_id in range(num_pairs)]
        num_phrases = [len(self.im2pairs[im_id]) for im_id in im_ids]

        max_phrases = max(num_phrases)

        gt_labels = np.zeros((self.batch_size, max_phrases, self.max_boxes),
                             dtype=np.float32)
        phrase_features = np.zeros((self.batch_size, max_phrases, self.phrase_feature_dim),
                                   dtype=np.float32)

        for pair_id in range(num_pairs):
            im_id = self.im_ids[self.pair_index[start_pair + pair_id]]
            features = np.array(self.data[im_id], np.float32)
            num_boxes = min(len(features), self.max_boxes)
            features = features[:num_boxes, :self.region_feature_dim]
            region_features[pair_id, :num_boxes, :] = features

            np.random.shuffle(self.im2pairs[im_id])
            num_phrase = num_phrases[pair_id]
            for i, sample_id in enumerate(self.im2pairs[im_id]):
                # paired image
                assert(self.pairs[0][sample_id] == im_id)
                
                # paired phrase
                phrase = self.pairs[1][sample_id]

                # phrase instance identifier
                p_id = self.pairs[2][sample_id]
                
                overlaps = np.array(self.data['%s_%s_%s' % (im_id, phrase, p_id)])
                
                # last 4 dimensions of overlaps are ground truth box coordinates
                assert(num_boxes <= len(overlaps) - 4)
                overlaps = overlaps[:num_boxes]
                phrase_features[pair_id, i, :] = self.w2v_dict[phrase]
                gt_labels[pair_id, i, :num_boxes] = overlaps >= self.success_thresh
                if self.is_train:
                    num_pos = int(np.sum(gt_labels[pair_id, :]))
                    num_neg = num_pos * self.neg_to_pos_ratio
                    negs = np.random.permutation(np.where(overlaps < 0.3)[0])
                
                    if len(negs) < num_neg: # if not enough negatives
                        negs = np.random.permutation(np.where(overlaps < 0.4)[0])

                    # logistic loss only counts a region labeled as -1 negative
                    gt_labels[pair_id, i, negs[:num_neg]] = -1

        feed_dict = {self.plh['phrase'] : phrase_features,
                     self.plh['region'] : region_features,
                     self.plh['train_phase'] : self.is_train,
                     self.plh['num_boxes'] : self.max_boxes,
                     self.plh['num_phrases'] : max_phrases,
                     self.plh['phrase_denom'] : np.sum(num_phrases).astype(np.float32) + 1e-6,
                     self.plh['labels'] : gt_labels
        }

        return feed_dict, gt_labels, num_phrases

