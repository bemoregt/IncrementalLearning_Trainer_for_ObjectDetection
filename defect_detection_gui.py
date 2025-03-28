import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone  # 변경
from torchvision.models.resnet import ResNet18_Weights  # 추가
from torchvision.transforms import transforms
import cv2
import numpy as np
from tkinterdnd2 import *
import os
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class CustomDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_width, image_height = image.size

        # YOLO 포맷 어노테이션 로드 및 변환
        boxes = []
        labels = []
        with open(self.annotation_paths[idx], 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # YOLO 포맷(normalized)을 Faster R-CNN 포맷(absolute)으로 변환
                x1 = (x_center - width/2) * image_width
                y1 = (y_center - height/2) * image_height
                x2 = (x_center + width/2) * image_width
                y2 = (y_center + height/2) * image_height
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # 결함은 클래스 1로 설정

        # 텐서로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    
    
    
class DefectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection with Faster R-CNN")
        
        self.threshold = tk.StringVar(value='0.5')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # 학습 데이터 저장용 리스트
        self.training_images = []
        self.training_annotations = []
        
        # 학습 손실 저장용 리스트
        self.losses = []
        
        self.create_gui_elements()
    
    def load_model(self):
        # ResNet18 백본 생성 (최신 방식)
        backbone = resnet_fpn_backbone(
            backbone_name='resnet18',
            weights=ResNet18_Weights.DEFAULT,
            trainable_layers=3
        )
        
        # Faster R-CNN 모델 생성
        model = FasterRCNN(
            backbone=backbone,
            num_classes=2,  # background + defect
            rpn_anchor_generator=None,  # 기본값 사용
            box_roi_pool=None,  # 기본값 사용
            rpn_head=None,  # 기본값 사용
            box_head=None,  # 기본값 사용
            box_predictor=None  # 기본값 사용
        )
        
        model.to(self.device)
        return model
        
    
    def update_loss_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('#333333')
        self.fig.patch.set_facecolor('#333333')
        
        if len(self.losses) > 0:
            self.ax.plot(range(1, len(self.losses) + 1), self.losses, 'b-', color='#00ff00')
            self.ax.set_xlim(1, 25)
            self.ax.set_ylim(0, max(self.losses)*1.1)  # y축 최소값을 0으로 설정
            
            # 축 레이블 추가 (폰트 크기 조정)
            self.ax.set_xlabel('Epoch', color='white', fontsize=8)
            self.ax.set_ylabel('Error', color='white', fontsize=8)
            
            # 격자 추가
            self.ax.grid(True, linestyle='--', alpha=0.3, color='white')
            
            # 축 눈금 색상과 크기 설정
            self.ax.tick_params(axis='x', colors='white', labelsize=8)
            self.ax.tick_params(axis='y', colors='white', labelsize=8)
            
            # 테두리 색상 설정
            for spine in self.ax.spines.values():
                spine.set_color('white')
        
        self.graph_canvas.draw()

    def create_gui_elements(self):
        # 메인 프레임 생성
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # 상단 컨트롤 프레임
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=5)
        
        # 버튼 프레임
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        # 이미지 로드 버튼
        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 학습 버튼 추가
        train_btn = tk.Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        # 임계값 선택 프레임
        threshold_frame = tk.Frame(control_frame)
        threshold_frame.pack(side=tk.LEFT, padx=20)
        
        # 임계값 라벨
        tk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
        
        # 임계값 콤보박스
        values = [str(round(i/10, 1)) for i in range(1, 10)]
        threshold_combo = ttk.Combobox(threshold_frame, values=values, 
                                    textvariable=self.threshold, width=5)
        threshold_combo.pack(side=tk.LEFT)
        threshold_combo.bind('<<ComboboxSelected>>', self.threshold_changed)
        
        # 이미지 표시 영역
        self.canvas = tk.Canvas(main_frame, width=640, height=480)
        self.canvas.pack(pady=10)
        
        # 드래그 앤 드롭 바인딩
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.drop_image)
        
        # 상태 표시줄
        self.status_label = tk.Label(self.root, text="Ready", bd=1, 
                                   relief=tk.SUNKEN, anchor=tk.W,
                                   bg='gray', height=2)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 그래프를 위한 프레임 추가
        graph_frame = tk.Frame(main_frame)
        graph_frame.pack(pady=5, side=tk.BOTTOM)
        
        # matplotlib 피규어 생성 부분 수정
        self.fig = Figure(figsize=(6, 2), dpi=100, facecolor='#333333')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#333333')

        # 여백 조정
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)  # 여백 줄이기
                
        # 캔버스에 피규어 추가
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack()
        
    def drop_image(self, event):
        # 드롭된 파일 경로들을 공백으로 분리
        file_paths = event.data.strip().split()
        print(f"Dropped file paths: {file_paths}")  # 디버그용 출력
        
        # 이미지 파일 경로 찾기
        image_path = None
        txt_path = None
        for path in file_paths:
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = path
            elif path.lower().endswith('.txt'):
                txt_path = path
        
        if image_path and txt_path:
            try:
                # 이미지 로드 확인
                image = Image.open(image_path)
                print(f"Image size: {image.size}")  # 디버그용 출력
                
                # 학습 데이터에 추가
                self.training_images.append(image_path)
                self.training_annotations.append(txt_path)
                
                # 현재 이미지 설정
                self.original_image = image
                self.display_image = self.original_image.copy()
                self.resize_image_for_display()
                
                # YOLO 바운딩 박스 표시
                image_width, image_height = self.original_image.size
                with open(txt_path, 'r') as f:
                    for line in f:
                        try:
                            class_id, x_center, y_center, width, height = map(float, line.strip().split())
                            
                            # YOLO 좌표를 픽셀 좌표로 변환
                            x1 = (x_center - width/2) * image_width
                            y1 = (y_center - height/2) * image_height
                            x2 = (x_center + width/2) * image_width
                            y2 = (y_center + height/2) * image_height
                            
                            # 캔버스 상의 좌표로 변환
                            canvas_x1 = int(x1 * self.scale_ratio) + self.x_offset
                            canvas_y1 = int(y1 * self.scale_ratio) + self.y_offset
                            canvas_x2 = int(x2 * self.scale_ratio) + self.x_offset
                            canvas_y2 = int(y2 * self.scale_ratio) + self.y_offset
                            
                            # 바운딩 박스 그리기
                            self.canvas.create_rectangle(
                                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                outline='green',  # 학습용 바운딩 박스는 녹색으로 표시
                                width=2
                            )
                            # 'GroundTruth' 텍스트 추가
                            self.canvas.create_text(
                                canvas_x1, max(canvas_y1-5, self.y_offset),
                                text='GroundTruth',
                                fill='green',
                                anchor=tk.SW
                            )
                            print(f"Drew box at: {canvas_x1}, {canvas_y1}, {canvas_x2}, {canvas_y2}")  # 디버그용 출력
                        except Exception as e:
                            print(f"Error processing annotation line: {e}")
                
                self.status_label.config(
                    text=f"Added training pair ({len(self.training_images)}): {os.path.basename(image_path)}")
            
            except Exception as e:
                print(f"Error loading image: {e}")
                self.status_label.config(text=f"Error loading image: {str(e)}")
        else:
            self.status_label.config(
                text="Please drop both image and its corresponding annotation file together")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image = self.original_image.copy()
            self.resize_image_for_display()
            self.detect_defects()
            self.status_label.config(text=f"Loaded image: {file_path}")

    def resize_image_for_display(self):
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width, img_height = self.original_image.size
        
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        self.scale_ratio = min(width_ratio, height_ratio)
        
        new_width = int(img_width * self.scale_ratio)
        new_height = int(img_height * self.scale_ratio)
        
        self.display_image = self.original_image.resize((new_width, new_height), 
                                                      Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        
        self.canvas.delete("all")
        
        # 이미지의 오프셋(캔버스 상의 시작 위치) 저장
        self.x_offset = x_center - new_width // 2
        self.y_offset = y_center - new_height // 2
        
        self.image_position = (
            self.x_offset,
            self.y_offset,
            self.x_offset + new_width,
            self.y_offset + new_height
        )
        
        self.canvas.create_image(
            x_center, 
            y_center,
            image=self.photo_image,
            anchor=tk.CENTER
        )

    def train_model(self):
        if len(self.training_images) == 0:
            self.status_label.config(text="No training data available!")
            return

        # 손실 리스트 초기화
        self.losses = []
        
        # 학습 모드로 전환
        self.model.train()

        # 데이터셋 및 데이터로더 생성
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        try:
            dataset = CustomDataset(
                self.training_images,
                self.training_annotations,
                transform=transform
            )
            
            data_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                collate_fn=lambda x: tuple(zip(*x))
            )

            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            
            num_epochs = 25
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for i, (images, targets) in enumerate(data_loader):
                    try:
                        images = list(image.to(self.device) for image in images)
                        targets = [{k: v.to(self.device) for k, v in t.items()} 
                                 for t in targets]

                        optimizer.zero_grad()
                        loss_dict = self.model(images, targets)
                        
                        losses = sum(loss for loss in loss_dict.values())
                        epoch_loss += losses.item()

                        losses.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        continue

                avg_loss = epoch_loss / len(data_loader)
                self.losses.append(avg_loss)
                
                # 그래프 업데이트
                self.update_loss_plot()
                
                self.status_label.config(
                    text=f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
                self.root.update()

            # 학습된 모델 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"trained_model_{timestamp}.pth"
            torch.save(self.model.state_dict(), model_path)
            
            self.model.eval()
            self.status_label.config(text=f"Training completed! Model saved as: {model_path}")
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            self.status_label.config(text=f"Training error: {str(e)}")

    def detect_defects(self):
        if not hasattr(self, 'original_image'):
            self.status_label.config(text="Please load an image first!")
            return
        
        # 추론 모드로 전환
        self.model.eval()  # 이 부분 추가
        
        self.resize_image_for_display()
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(self.original_image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        self.visualize_predictions(predictions[0])

    def threshold_changed(self, event=None):
        if hasattr(self, 'original_image'):
            self.detect_defects()

    def visualize_predictions(self, prediction):
        if not hasattr(self, 'image_position'):
            return
            
        x_start, y_start, x_end, y_end = self.image_position
        display_width = x_end - x_start
        display_height = y_end - y_start
        
        orig_width, orig_height = self.original_image.size
        
        width_ratio = display_width / orig_width
        height_ratio = display_height / orig_height
        
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        threshold = float(self.threshold.get())
        
        valid_detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                x1 = int(box[0] * width_ratio) + x_start
                y1 = int(box[1] * height_ratio) + y_start
                x2 = int(box[2] * width_ratio) + x_start
                y2 = int(box[3] * height_ratio) + y_start
                
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline='red',
                    width=2
                )
                
                # 점수와 'Prediction' 텍스트를 함께 표시
                self.canvas.create_text(
                    x1, max(y1-5, y_start),
                    text=f'Prediction ({score:.2f})',
                    fill='red',
                    anchor=tk.SW
                )
                valid_detections.append((x1, y1, x2, y2, score))
        
        self.status_label.config(
            text=f"Found {len(valid_detections)} detections (threshold: {threshold})")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = DefectDetectionGUI(root)
    root.mainloop()