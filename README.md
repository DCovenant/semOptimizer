# Smart Traffic Signal Optimizer

Sistema de otimização de semáforos baseado em visão computacional. Usa uma câmara montada na interseção para detetar veículos por direção e decidir, em tempo real, se o ciclo do semáforo deve ser ajustado.

## O Problema

Em muitas interseções, os semáforos operam com ciclos fixos. Isto gera situações em que um lado tem 10+ carros parados no vermelho enquanto o lado verde está completamente vazio — trânsito desnecessário.

## A Solução

Uma câmara no poste do semáforo alimenta dois modelos de ML que, em conjunto, detetam o volume de tráfego por direção e sugerem ajustes ao ciclo.

## Arquitetura

O sistema opera em duas fases distintas:

```
┌─────────────────────────────────────────────────────────┐
│                    FASE 1 — STARTUP                     │
│                  (corre UMA vez)                        │
│                                                         │
│   Câmara → SegFormer → Máscara da estrada → Contornos  │
│                                    │                    │
│                                    ▼                    │
│                          Zonas automáticas              │
│                     (N / S / E / W guardadas)           │
│                                    │                    │
│                                    ▼                    │
│                        road_config.json                 │
└────────────────────────────────────┬────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────┐
│                FASE 2 — LOOP CONTÍNUO                   │
│             (corre frame a frame)                       │
│                                                         │
│   Câmara → YOLOv8n → Veículos detetados                │
│                            │                            │
│                            ▼                            │
│              Cruzar posições com zonas                  │
│              (carregadas do config)                     │
│                            │                            │
│                            ▼                            │
│               Contagem por direção                      │
│            (red_count vs green_count)                   │
│                            │                            │
│                            ▼                            │
│                Lógica de decisão                        │
│          "Manter ciclo" ou "Trocar sinal"               │
└─────────────────────────────────────────────────────────┘
```

### Porquê duas fases?

O layout da estrada não muda. Correr segmentação semântica (SegFormer) a cada frame seria um desperdício de recursos. Ao separá-lo num passo de inicialização:

- **Fase 1** corre uma vez (~2-5s, mesmo em CPU) e guarda o resultado em disco
- **Fase 2** corre apenas YOLO, que é muito mais leve (~100-500ms por frame em CPU)
- Num Raspberry Pi, isto faz a diferença entre viável e inviável
- Se a câmara for movida ou a estrada mudar, basta re-correr a Fase 1

### Recalibração

A Fase 1 deve ser re-executada quando:
- A câmara é movida ou reposicionada
- Há obras na estrada que alterem o layout
- As condições mudam drasticamente (neve a cobrir marcações)

Pode ser agendada para correr periodicamente (ex: 1x por dia de madrugada) como safety net.

## Stack Técnica

| Componente | Tecnologia | Tamanho | Velocidade (CPU) |
|---|---|---|---|
| Segmentação da estrada | SegFormer-B0 (Cityscapes) | ~14MB | ~2-5s por imagem |
| Deteção de veículos | YOLOv8n (COCO) | ~6MB | ~100-500ms por frame |
| Processamento de imagem | OpenCV | - | - |
| Inferência otimizada | ONNX Runtime | - | 2-3x mais rápido que PyTorch |

## Estrutura do Projeto

```
traffic-signal-optimizer/
├── README.md
├── config/
│   └── road_config.json          # zonas geradas pela Fase 1
├── notebooks/
│   ├── road_edge_detection.ipynb  # experimenta Fase 1 interativamente
│   └── traffic_signal_optimizer.ipynb  # experimenta Fase 2
├── src/
│   ├── phase1_road_setup.py      # SegFormer → detetar estrada → gerar zonas
│   ├── phase2_vehicle_loop.py    # YOLO → contar carros → decisão
│   ├── signal_logic.py           # lógica de decisão do semáforo
│   └── visualize.py              # funções de visualização e debug
├── models/
│   ├── yolov8n.onnx              # YOLO exportado para ONNX
│   └── segformer-b0.onnx         # SegFormer exportado para ONNX
└── tests/
    └── test_with_sample_images/
```

## road_config.json (output da Fase 1)

```json
{
  "image_size": [1920, 1080],
  "road_mask_path": "config/road_mask.npy",
  "zones": {
    "north": {"bbox": [400, 0, 800, 300], "direction": "incoming"},
    "south": {"bbox": [400, 700, 800, 1080], "direction": "incoming"},
    "east":  {"bbox": [1100, 300, 1920, 700], "direction": "incoming"},
    "west":  {"bbox": [0, 300, 400, 700], "direction": "incoming"}
  },
  "road_contours_path": "config/road_contours.npy",
  "calibration_date": "2026-05-13",
  "camera_moved_since_calibration": false
}
```

## Hardware — Deploy em Raspberry Pi

### Setup recomendado

**Raspberry Pi 5 (8GB) + AI HAT+ 26 TOPS**

O RPi 5 de 8GB dá espaço suficiente para correr o SegFormer na Fase 1 sem problemas. A versão de 4GB fica apertada com PyTorch em memória; a de 16GB seria overkill para este caso.

O AI HAT+ (original, 26 TOPS) é o acelerador recomendado. Usa o chip Hailo-8 ligado ao PCIe do RPi 5, entregando inferência dedicada on-device sem necessidade de cloud. Com o HAT+, o YOLOv8n corre a ~60 FPS — a diferença entre um protótipo a 1-3 FPS e um sistema funcional em tempo real.

### Porquê o AI HAT+ e não o AI HAT+ 2?

O AI HAT+ 2 (40 TOPS, 8GB RAM dedicada, ~€130) foi desenhado para LLMs e modelos generativos. Para modelos de visão como YOLO e segmentação, a performance de computer vision é praticamente equivalente à do HAT+ original (~€70). Não compensa pagar quase o dobro por capacidades que este projeto não usa.

### O que NÃO usar

- **AI HAT+ 2** — overkill, desenhado para LLMs, não para CV
- **Google Coral USB** — mais antigo e menos integrado com o ecossistema RPi
- **RPi 4** — sem PCIe, não suporta AI HATs

### Bill of Materials

| Componente | Preço ~€ |
|---|---|
| Raspberry Pi 5 (8GB) | 85 |
| AI HAT+ 26 TOPS | 70 |
| Câmara RPi v3 | 30 |
| Alimentação USB-C 5V/5A | 15 |
| Cooler ativo oficial | 10 |
| MicroSD A2 ou NVMe | 15-30 |
| **Total** | **~225-240** |

### Deploy em produção

1. **No PC de desenvolvimento:** exportar ambos os modelos para ONNX
   ```bash
   # YOLO
   yolo export model=yolov8n.pt format=onnx
   # SegFormer — usar optimum da Hugging Face
   optimum-cli export onnx --model nvidia/segformer-b0-finetuned-cityscapes-1024-1024 segformer-b0-onnx/
   ```
2. **No RPi:** instalar apenas `onnxruntime` (sem PyTorch/Transformers — poupa ~2GB de RAM)
3. **Fase 1** corre no primeiro boot ou quando recalibração é necessária (~2-5s)
4. **Fase 2** corre em loop contínuo — ~60 FPS com AI HAT+, ~1-3 FPS sem
5. A câmara RPi v3 liga diretamente ao conector CSI do RPi 5, por baixo do AI HAT+

### Performance esperada

| Cenário | FPS estimado |
|---|---|
| RPi 5 (CPU only, PyTorch) | ~1-2 FPS |
| RPi 5 (CPU only, ONNX) | ~3-5 FPS |
| RPi 5 + AI HAT+ 26 TOPS | ~60 FPS |

## Lógica de Decisão

A decisão de trocar o sinal é baseada em regras simples (v1):

- Se `red_count > 0` e `green_count == 0` → trocar imediatamente
- Se `red_count - green_count >= threshold` → sugerir troca
- Caso contrário → manter ciclo atual

Melhorias futuras:
- **Temporal smoothing** — exigir N frames consecutivos a concordar antes de trocar
- **Tracking** — usar `model.track()` para distinguir carros parados de carros em movimento
- **Tempo mínimo de verde** — nunca trocar antes de X segundos (segurança)
- **Prioridade** — dar mais peso a autocarros/veículos de emergência
- **RL** — substituir regras fixas por reinforcement learning treinado no SynTraC

## Datasets Úteis

- **SynTraC** — dataset sintético (CARLA) para controlo de semáforos com RL, 86K+ imagens
- **UA-DETRAC** — 140K frames reais de tráfego com 1.21M bounding boxes
- **Cityscapes** — segmentação urbana (o que treinou o SegFormer)
- **BDD100K** — 100K vídeos de condução com segmentação

## Geração de Dados Sintéticos (CARLA + Augmentations)

Os modelos (YOLOv8, SegFormer) já vêm pré-treinados, mas para validar e fazer fine-tuning em cenários de interseção, podemos gerar dados sintéticos ilimitados com o simulador CARLA.

### Condições meteorológicas no CARLA

O CARLA expõe parâmetros independentes via Python API, permitindo criar cenários variados:

```python
import carla

weather = carla.WeatherParameters(
    cloudiness=90.0,              # nebulosidade (0-100%)
    precipitation=80.0,           # chuva (0-100%)
    precipitation_deposits=60.0,  # poças no chão (0-100%)
    wind_intensity=70.0,          # vento (0-100%)
    fog_density=50.0,             # nevoeiro densidade (0-100%)
    fog_distance=10.0,            # nevoeiro distância (metros)
    wetness=100.0,                # estrada molhada (0-100%)
    sun_altitude_angle=-30.0      # noite (< 0 = abaixo do horizonte)
)
world.set_weather(weather)
```

Presets disponíveis: ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, MidRainyNoon, HardRainNoon, SoftRainNoon, ClearSunset, CloudySunset, WetSunset, HardRainSunset, SoftRainSunset.

O modo noturno ativa-se automaticamente quando `sun_altitude_angle < 0`, acendendo luzes de rua e de veículos.

### Pós-processamento com Augmentations

O CARLA não simula artefactos de câmara (grain, motion blur). Para isso, aplicar augmentations em pós-processamento com Albumentations:

```python
import albumentations as A

augment = A.Compose([
    A.GaussNoise(var_limit=(10, 50)),                          # grain / ruído de sensor
    A.MotionBlur(blur_limit=7),                                 # motion blur
    A.RandomBrightnessContrast(p=0.5),                          # variação de luz
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),        # nevoeiro extra
    A.RandomSunFlare(src_radius=100, p=0.3),                    # reflexos de sol
    A.ImageCompression(quality_lower=40, quality_upper=80),      # compressão JPEG (câmara barata)
])

augmented = augment(image=frame)["image"]
```

### Pipeline de geração

A combinação CARLA + Albumentations cobre praticamente todas as condições reais:

| Condição | Fonte |
|---|---|
| Chuva, nevoeiro, noite, pôr-do-sol | CARLA (nativo) |
| Poças, estrada molhada, vento | CARLA (nativo) |
| Grain / ruído de sensor | Albumentations (pós) |
| Motion blur, desfoque | Albumentations (pós) |
| Reflexos de sol / lens flare | Albumentations (pós) |
| Compressão de câmara barata | Albumentations (pós) |
| Variações de brilho / contraste | Albumentations (pós) |

Isto permite gerar datasets grandes e variados sem sair de casa, com ground truth perfeito (bounding boxes, segmentação) gerado automaticamente pelo simulador.

## Otimização para Microcontroladores (TinyML)

O objetivo final é espremer o modelo ao máximo para correr num microcontrolador pequeno e barato. O treino acontece na máquina de desenvolvimento (Ryzen 5 7500F + RX 9060 XT 16GB via ROCm), mas o deploy é num MCU de <€20.

### Filosofia: Treinar grande, comprimir ao máximo

O pipeline de otimização segue uma cadeia de compressão progressiva. Cada etapa reduz o modelo mantendo o máximo de precisão possível.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MÁQUINA DE DESENVOLVIMENTO                   │
│                  (Ryzen 5 7500F + RX 9060 XT)                   │
│                                                                 │
│   1. TREINAR PROFESSOR                                          │
│      YOLOv8n completo → modelo grande mas preciso               │
│      ~6MB, FP32, ~3.2M parâmetros                               │
│                          │                                      │
│                          ▼                                      │
│   2. KNOWLEDGE DISTILLATION                                     │
│      Treinar "aluno" que imita o professor                      │
│      MobileNetV3-Small ou custom CNN                            │
│      ~500KB-1MB, FP32, ~100-500K parâmetros                     │
│                          │                                      │
│                          ▼                                      │
│   3. PRUNING                                                    │
│      Remover neurónios e conexões que contribuem pouco           │
│      Redução de 50-80% dos pesos com <5% perda de precisão      │
│      ~200-500KB                                                 │
│                          │                                      │
│                          ▼                                      │
│   4. QUANTIZAÇÃO                                                │
│      FP32 (32 bits) → INT8 (8 bits)                             │
│      Redução ~4x em tamanho                                     │
│      ~50-150KB                                                  │
│                          │                                      │
│                          ▼                                      │
│   5. EXPORTAR                                                   │
│      Converter para TFLite Micro (.tflite)                      │
│      Modelo final: ~50-150KB, INT8, pronto para MCU             │
│                                                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MICROCONTROLADOR                           │
│                  (ESP32-S3 ou STM32H7)                           │
│                                                                 │
│   Flash do modelo TFLite Micro                                  │
│   Inferência em INT8 → decisão em <100ms                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Etapa 1 — Treinar o Professor (YOLOv8n)

O modelo "professor" é o YOLOv8n treinado/fine-tuned na máquina de desenvolvimento. É grande demais para um MCU, mas serve como referência de precisão e para gerar labels automaticamente.

```python
from ultralytics import YOLO

# Treinar na RX 9060 XT via ROCm
# Instalar: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
model = YOLO("yolov8n.pt")
model.train(data="intersection_dataset.yaml", epochs=100, imgsz=640, device=0)
```

O professor gera "soft labels" — em vez de "carro" ou "não carro", produz probabilidades como "92% carro, 5% camião, 3% fundo". Estas probabilidades contêm informação rica sobre o que o modelo aprendeu.

### Etapa 2 — Knowledge Distillation

O aluno é um modelo muito mais pequeno que aprende a imitar as probabilidades do professor, não os dados originais. Isto funciona melhor do que treinar o aluno diretamente porque as soft labels do professor codificam relações entre classes que os labels "duros" (0 ou 1) não capturam.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCarCounter(nn.Module):
    """
    Modelo aluno mínimo para classificar: 0, 1-3, 4+ carros por zona.
    Arquitetura: entrada → 3 camadas conv → global avg pool → 3 classes
    ~50-200K parâmetros (vs 3.2M do YOLOv8n)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 3)  # 3 classes: 0, 1-3, 4+

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.7):
    """
    Combina dois sinais de aprendizagem:
    - soft_loss: imitar as probabilidades do professor (knowledge transfer)
    - hard_loss: acertar nos labels reais (ground truth)
    alpha controla o peso relativo: mais alto = mais foco no professor
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

A chave aqui é a **simplificação do problema**. O YOLOv8n faz deteção de objetos completa (bounding boxes + classes). O aluno faz apenas classificação: dado um crop de uma zona da interseção, quantos carros lá estão? Isto reduz a complexidade em ordens de magnitude.

### Etapa 3 — Pruning (Poda)

Depois do distillation, muitos neurónios no modelo aluno contribuem pouco para o resultado final. Pruning remove-os.

```python
import torch.nn.utils.prune as prune

# Pruning não-estruturado — remove pesos individuais (mais flexível)
for name, module in student_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.5)  # remove 50%

# Pruning estruturado — remove filtros inteiros (mais eficiente em hardware)
for name, module in student_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)

# Tornar o pruning permanente (remover a máscara e reduzir o modelo)
for name, module in student_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.remove(module, "weight")
```

Tipos de pruning e trade-offs:

| Tipo | O que remove | Redução típica | Impacto na precisão |
|---|---|---|---|
| Não-estruturado | Pesos individuais (zeros espalhados) | 50-90% dos pesos | Baixo |
| Estruturado | Filtros/canais inteiros | 30-70% dos filtros | Médio |
| Iterativo | Ciclos de prune → retrain → prune | Máximo possível | Controlado |

O pruning iterativo é o mais eficaz: poda 20% → retrain 10 epochs → poda mais 20% → retrain → repete. Em cada ciclo, o modelo readapta-se à nova estrutura.

### Etapa 4 — Quantização

Converter de FP32 (vírgula flutuante, 32 bits) para INT8 (inteiro, 8 bits). Os MCUs são muito mais rápidos com aritmética inteira.

```python
import tensorflow as tf

# Converter PyTorch → ONNX → TFLite (caminho mais robusto para MCUs)

# 1. Exportar para ONNX
torch.onnx.export(student_model, dummy_input, "student.onnx", opset_version=13)

# 2. Converter ONNX → TFLite com quantização INT8
# (usar onnx2tf ou ai-edge-torch)

# Quantização post-training com dataset de calibração
converter = tf.lite.TFLiteConverter.from_saved_model("student_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dataset de calibração — o quantizador precisa de amostras reais
# para calcular os ranges de ativação de cada camada
def representative_dataset():
    for image in calibration_images[:100]:
        yield [image.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Forçar INT8 completo (sem fallback para float)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("student_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Tamanho final: {len(tflite_model) / 1024:.1f} KB")
```

Impacto da quantização:

| Precisão | Tamanho por peso | Modelo 200K params | Velocidade MCU |
|---|---|---|---|
| FP32 | 4 bytes | ~800 KB | Baseline |
| FP16 | 2 bytes | ~400 KB | ~1.5x |
| INT8 | 1 byte | ~200 KB | ~3-4x |
| INT4 | 0.5 bytes | ~100 KB | ~5-6x (limitado) |

### Etapa 5 — Deploy no Microcontrolador

O modelo TFLite Micro final é compilado e flashado no MCU como um array C.

```c
// Exemplo conceitual para ESP32-S3 ou STM32H7
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "student_int8_model.h"  // modelo convertido para array C

// Alocar memória para o interpretador (~50-200KB)
constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Carregar modelo
const tflite::Model* model = tflite::GetModel(student_int8_tflite);

// Criar interpretador
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
interpreter.AllocateTensors();

// Inferência
// 1. Capturar frame da câmara
// 2. Redimensionar para input do modelo (ex: 96x96 ou 128x128)
// 3. Copiar pixels para o tensor de entrada
memcpy(interpreter.input(0)->data.int8, image_data, input_size);

// 4. Correr inferência
interpreter.Invoke();

// 5. Ler resultado (0=vazio, 1=poucos carros, 2=muitos carros)
int8_t* output = interpreter.output(0)->data.int8;
int predicted_class = argmax(output, 3);
```

### Hardware MCU — Opções

| MCU | Preço ~€ | RAM | Flash | Câmara | TinyML Support |
|---|---|---|---|---|---|
| ESP32-S3 | 8-12 | 512KB | 8MB | OV2640 (~€3) | TFLite Micro, ESP-DL |
| STM32H747 | 15-25 | 1MB | 2MB | DCMI interface | STM32Cube.AI (auto-otimiza) |
| Arduino Nicla Vision | 60-70 | 1MB | 2MB | Integrada 2MP | OpenMV + TFLite |
| Kendryte K210 | 8-10 | 8MB SRAM | 16MB | DVP interface | NNCASE, MaixPy |

Para o caso mais barato e funcional: **ESP32-S3 + OV2640** (~€12 total). O ESP32-S3 tem aceleração vetorial para INT8, WiFi/BT integrado (para enviar decisões ao controlador do semáforo), e uma comunidade enorme.

Para a melhor experiência de desenvolvimento: **STM32H747** com STM32Cube.AI. O Cube.AI pega no modelo TFLite/ONNX e gera automaticamente código C otimizado para o chip, com análise de memória e performance antes do deploy.

### Simplificação do Problema para MCU

O modelo no MCU **não** faz o mesmo que o YOLOv8. O truque é reformular o problema:

```
YOLOv8 no PC:        Imagem → [lista de bounding boxes + classes + confidence]
                     Complexo, ~6MB, precisa de GPU

Modelo no MCU:       Crop de uma zona → "0 carros" | "1-3 carros" | "4+ carros"
                     Classificação simples, ~100KB, corre em INT8 no ESP32
```

Em vez de uma câmara inteligente que faz tudo, o MCU faz uma tarefa simples por zona. Podes até ter múltiplos ESP32 (um por zona, ~€12 cada) a reportar a um controlador central que faz a lógica de decisão.

### Pipeline de geração de labels

O professor (YOLOv8) gera os labels para o aluno automaticamente:

```python
# 1. Correr YOLOv8 em todas as imagens de treino
teacher = YOLO("best_teacher.pt")

# 2. Para cada imagem, contar carros por zona e gerar label
labels = []
for img_path in training_images:
    results = teacher(img_path, verbose=False)[0]
    car_count = sum(1 for box in results.boxes if int(box.cls[0]) in VEHICLE_CLASSES)

    if car_count == 0:
        label = 0     # vazio
    elif car_count <= 3:
        label = 1     # poucos
    else:
        label = 2     # muitos

    labels.append((img_path, label))

# 3. Treinar o aluno com estes labels
#    (+ distillation com soft labels do professor)
```

### Métricas de sucesso

O modelo está pronto para deploy quando:

| Métrica | Target |
|---|---|
| Tamanho do modelo (.tflite) | < 150 KB |
| RAM necessária (tensor arena) | < 200 KB |
| Precisão (accuracy) | > 85% nas 3 classes |
| Tempo de inferência no MCU | < 100ms |
| Consumo energético | < 500mW em inferência |

### Ferramentas úteis

- **Netron** — visualizar arquiteturas de modelos (ONNX, TFLite, PyTorch). Essencial para perceber o que estás a comprimir.
- **STM32Cube.AI** — analisa o modelo antes do deploy: mostra RAM/Flash necessária, ops por camada, estimativa de latência.
- **Edge Impulse** — plataforma web para treinar e fazer deploy de TinyML. Bom para prototipar rápido antes de fazer tudo manual.
- **ONNX Runtime Mobile** — alternativa ao TFLite, mais fácil de converter de PyTorch.
- **ai-edge-torch** — ferramenta do Google para converter PyTorch → TFLite diretamente.

## TODO

- [ ] Implementar `phase1_road_setup.py` com geração automática de zonas a partir dos contornos
- [ ] Implementar `phase2_vehicle_loop.py` com loop de câmara
- [ ] Exportar modelos para ONNX
- [ ] Testar em Raspberry Pi 5
- [ ] Adicionar temporal smoothing à lógica de decisão
- [ ] Adicionar tracking para distinguir carros parados vs em movimento
- [ ] Interface web para monitorização (FastAPI + Vue.js)
- [ ] Explorar RL com SynTraC como alternativa às regras fixas
- [ ] Montar pipeline de geração de dados no CARLA (interseção + câmara no poste)
- [ ] Gerar dataset com variações meteorológicas (sol, chuva, nevoeiro, noite)
- [ ] Aplicar augmentations (grain, blur, compressão) ao dataset gerado
- [ ] Definir e treinar arquitetura do modelo aluno (TinyCarCounter)
- [ ] Implementar pipeline de knowledge distillation (professor → aluno)
- [ ] Aplicar pruning iterativo com retrain
- [ ] Quantizar modelo para INT8 com dataset de calibração
- [ ] Exportar para TFLite Micro
- [ ] Testar inferência no ESP32-S3 com câmara OV2640
- [ ] Benchmarkar: tamanho < 150KB, latência < 100ms, precisão > 85%
- [ ] Comparar STM32Cube.AI vs TFLite Micro para deploy final
