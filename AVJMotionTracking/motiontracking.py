from AVJMotionTracking.deep_sort.application_util import preprocessing
from AVJMotionTracking.deep_sort.application_util import visualization
from AVJMotionTracking.deep_sort import nn_matching
from AVJMotionTracking.deep_sort.detection import Detection
from AVJMotionTracking.deep_sort.tracker import Tracker
from AVJMotionTracking.data_loader import *

def AVJ_motion_tracking(path_CMRimages, path_detections, path_trackresults):
    for seq in os.listdir(path_CMRimages):
        sequence_dir = os.path.join(path_CMRimages,seq)
        detection_file = os.path.join(path_detections, seq+".csv")
        min_confidence = 0.3
        nn_budget = 100
        display = False
        output_file = os.path.join(path_trackresults, seq+".csv")
        nms_max_overlap = 1.0
        min_detection_height = 0
        max_cosine_distance = 0.2

        tracking(sequence_dir, detection_file, output_file,
            min_confidence, nms_max_overlap, min_detection_height,
            max_cosine_distance, nn_budget, display)


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def tracking(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):

    seq_info = load_data(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


if __name__ == '__main__':
    import time
    start = time.time()

    path_CMRimages = "../datasets/CMRimages/testingset"
    path_groundtruth = "../datasets/Groundtruth"
    path_detections = "../datasets/DetectionResults"
    path_trackresults = "../datasets/TrackResults"

    AVJ_motion_tracking(path_CMRimages, path_detections, path_trackresults)

    end = time.time()
    print ('execution time (model prediction): ', (end - start))