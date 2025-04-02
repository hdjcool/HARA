# pose_estimator.py
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 roi_padding=40):
        """
        MediaPipe 포즈 추정기 초기화 (동적 ROI 지원)
        Args:
            min_detection_confidence (float): 최소 감지 신뢰도
            min_tracking_confidence (float): 최소 추적 신뢰도
            roi_padding (int): 감지된 랜드마크 주변 ROI 여유 공간 (픽셀)
        """
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False, # 비디오 스트림 처리
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False, # 필요 시 True로 변경 가능
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.roi_bbox = None  # 현재 ROI 영역 [x_min, y_min, x_max, y_max]
        self.roi_active = False # 현재 ROI를 사용 중인지 여부
        self.roi_padding = roi_padding # 픽셀 단위 패딩

        print("PoseEstimator with Dynamic ROI initialized.")

    def _calculate_roi_from_landmarks(self, landmarks, frame_width, frame_height):
        """
        주어진 랜드마크(Full-frame Normalized)를 기반으로 ROI BBox(픽셀 좌표)를 계산합니다.
        Args:
            landmarks (list): MediaPipe Landmark 객체 리스트.
            frame_width (int): 원본 프레임 너비.
            frame_height (int): 원본 프레임 높이.
        Returns:
            tuple: 계산된 ROI 좌표 (x_min, y_min, x_max, y_max) 픽셀. 실패 시 None.
        """
        if not landmarks:
            return None

        # Normalized 좌표를 픽셀 좌표로 변환
        try:
            x_coords = [lm.x * frame_width for lm in landmarks]
            y_coords = [lm.y * frame_height for lm in landmarks]

            if not x_coords or not y_coords: # 유효 좌표 없으면 실패
                 return None

        except AttributeError: # 혹시 landmarks 구조가 다를 경우
             print("Error: Landmarks object structure unexpected.")
             return None


        # 바운딩 박스 계산
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # 패딩 추가
        x_min = int(x_min - self.roi_padding)
        y_min = int(y_min - self.roi_padding)
        x_max = int(x_max + self.roi_padding)
        y_max = int(y_max + self.roi_padding)

        # 프레임 경계 내로 제한
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame_width, x_max)
        y_max = min(frame_height, y_max)

        # 유효한 박스인지 확인 (넓이/높이가 0보다 커야 함)
        if x_min < x_max and y_min < y_max:
            return (x_min, y_min, x_max, y_max)
        else:
            return None

    def estimate_pose(self, frame):
        """
        동적 ROI 기반 포즈 추정
        Args:
            frame (numpy.ndarray): 입력 프레임 (BGR)
        Returns:
            tuple: (pose_landmarks, annotated_frame)
                   pose_landmarks: MediaPipe 포즈 랜드마크 객체 (전체 프레임 기준 Normalized). 감지 실패 시 None.
                   annotated_frame: ROI와 포즈(감지 시)가 그려진 프레임 (BGR)
        """
        frame_height, frame_width, _ = frame.shape
        output_frame = frame.copy() # 결과용 프레임 복사
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ROI용 이미지 초기 설정
        input_image = frame_rgb
        offset_x, offset_y = 0, 0
        process_in_roi = False

        if self.roi_active and self.roi_bbox:
            x1, y1, x2, y2 = self.roi_bbox
            # 프레임 범위 내로 ROI 좌표 조정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)

            if x1 < x2 and y1 < y2:
                try:
                    # ROI 영역 잘라내기
                    input_image = frame_rgb[y1:y2, x1:x2].copy()
                    offset_x, offset_y = x1, y1 # 원본 프레임 기준 오프셋 저장
                    process_in_roi = True
                except Exception as e:
                     print(f"Error cropping ROI: {e}. Processing full frame instead.")
                     input_image = frame_rgb # 에러 시 전체 프레임으로 복귀
                     self.roi_active = False # ROI 비활성화
                     self.roi_bbox = None
            else:
                self.roi_active = False
                self.roi_bbox = None
                input_image = frame_rgb # 전체 프레임 사용

        # MediaPipe 포즈 처리
        results = None
        try:
            # 성능 향상을 위해 이미지 쓰기 불가 설정
            input_image.flags.writeable = False
            results = self.pose.process(input_image)
            input_image.flags.writeable = True
        except Exception as e:
            print(f"Error during MediaPipe pose processing: {e}")
            # 에러 발생 시 ROI 비활성화하여 다음 프레임은 전체 처리 시도
            self.roi_active = False
            self.roi_bbox = None


        # 결과 처리 및 다음 ROI 계산
        corrected_landmarks_object = None # 반환할 랜드마크 보정 객체

        if results and results.pose_landmarks:
            roi_landmarks = results.pose_landmarks # ROI 기준 랜드마크
            roi_h, roi_w = input_image.shape[:2] # 처리된 이미지(ROI 또는 전체)의 크기

            # 랜드마크 좌표 보정
            for landmark in roi_landmarks.landmark:
                # 1. ROI 내 픽셀 좌표 계산
                pixel_x_roi = landmark.x * roi_w
                pixel_y_roi = landmark.y * roi_h
                # 2. 전체 프레임 내 픽셀 좌표 계산
                pixel_x_frame = pixel_x_roi + offset_x
                pixel_y_frame = pixel_y_roi + offset_y
                # 3. 전체 프레임 기준 Normalized 좌표로 변환
                landmark.x = pixel_x_frame / frame_width
                landmark.y = pixel_y_frame / frame_height
                # landmark.z = landmark.z * roi_w  # 너비 기준 z좌표 보정 (선택사항)

            corrected_landmarks_object = roi_landmarks # 보정된 랜드마크 객체 저장

            # 다음 프레임을 위한 새로운 ROI 계산
            # 보정된 랜드마크(전체 프레임 Normalized) 사용
            new_roi = self._calculate_roi_from_landmarks(corrected_landmarks_object.landmark, frame_width, frame_height)

            if new_roi:
                self.roi_bbox = new_roi
                self.roi_active = True
            else:
                # 랜드마크는 감지했지만 ROI 계산 실패한 경우
                self.roi_active = False
                self.roi_bbox = None

        else:
            # 랜드마크 감지 실패
            if process_in_roi:
                self.roi_active = False
                self.roi_bbox = None

        # 시각화
        # 1. 현재 ROI 영역 그리기 (감지된 경우, 파란색 박스)
        if self.roi_active and self.roi_bbox:
             x1, y1, x2, y2 = self.roi_bbox
             cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif not self.roi_active: # 비활성 상태 표시
             cv2.putText(output_frame, "ROI Inactive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # 2. 포즈 랜드마크 그리기 (감지된 경우, 초록색 점, 빨간색 선)
        if corrected_landmarks_object:
            mp.solutions.drawing_utils.draw_landmarks(
                output_frame,
                corrected_landmarks_object,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2))

        return corrected_landmarks_object, output_frame

    def close(self):
        """MediaPipe 리소스 해제"""
        self.pose.close()
        print("MediaPipe Pose resources released.")


class PoseEstimator3D(PoseEstimator): # PoseEstimator를 상속받아 중복 최소화
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 roi_padding=40):
        """
        MediaPipe 3D 포즈 추정기 초기화 (동적 ROI 지원)
        PoseEstimator의 기능을 상속받습니다.
        Args:
            min_detection_confidence (float): 최소 감지 신뢰도
            min_tracking_confidence (float): 최소 추적 신뢰도
            roi_padding (int): 감지된 랜드마크 주변 ROI 여유 공간 (픽셀)
        """
        # PoseEstimator의 __init__ 호출
        super().__init__(min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence,
                         roi_padding=roi_padding)
        print("PoseEstimator3D with Dynamic ROI initialized (inherits from PoseEstimator).")

    def extract_3d_keypoints(self, landmarks, frame):
        """
        추정된 랜드마크(Full-frame Normalized)에서 3D 키포인트(픽셀 좌표 + 상대 깊이)를 추출합니다.
        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                       estimate_pose에서 반환된 보정된 랜드마크 객체.
            frame (numpy.ndarray): 원본 입력 프레임 (높이, 너비 정보 사용).
        Returns:
            list: 3D 키포인트 좌표 [(x_pixel, y_pixel, z_relative), ...]
                  랜드마크가 없으면 빈 리스트 반환.
        """
        if not landmarks:
            return []

        height, width, _ = frame.shape
        keypoints_3d = []

        for landmark in landmarks.landmark:
            # 2D 좌표 변환
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # 깊이 추정 (MediaPipe의 z 값 사용)
            # landmark.z는 MediaPipe에서 제공하는 상대적 깊이 값
            z = int(landmark.z * width)  # 깊이를 너비 기준으로 스케일링

            keypoints_3d.append((x, y, z))

        return keypoints_3d