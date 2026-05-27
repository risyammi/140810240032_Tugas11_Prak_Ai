import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ChromaLens · Color Palette Extractor",
    page_icon="🎨",
    layout="centered",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f11;
    color: #f0ede8;
  }

  /* ── Hero header ── */
  .hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
  }
  .hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
  }
  .hero p {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
  }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: #1a1a1f;
    border: 1.5px dashed #333;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #ff6b6b;
  }

  /* ── Swatch cards ── */
  .swatch-card {
    background: #1a1a1f;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
    transition: transform 0.2s;
  }
  .swatch-card:hover {
    transform: translateY(-3px);
  }
  .swatch-block {
    height: 90px;
    width: 100%;
  }
  .swatch-info {
    padding: 0.75rem;
    text-align: center;
  }
  .swatch-hex {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .swatch-rgb {
    font-size: 0.72rem;
    color: #666;
    margin-top: 2px;
  }
  .swatch-pct {
    font-size: 0.75rem;
    color: #aaa;
    margin-top: 4px;
  }

  /* ── Section label ── */
  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
  }

  /* ── K-Means detail box ── */
  .kmeans-box {
    background: #1a1a1f;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid;
  }
  .kmeans-iter {
    font-size: 0.8rem;
    color: #666;
  }

  /* ── Progress bar override ── */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #ff6b6b, #ffd93d) !important;
    border-radius: 99px;
  }

  /* ── Slider ── */
  [data-testid="stSlider"] {
    padding-top: 0.25rem;
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0f0f11; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_text_color(bg_rgb):
    """Return black or white depending on background luminance."""
    r, g, b = [x / 255.0 for x in bg_rgb]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0f0f11" if luminance > 0.45 else "#f0ede8"


def extract_colors(image: Image.Image, n_colors: int = 5):
    """
    Run K-Means on the image pixels and return:
        colors       – (n_colors, 3) array of RGB centroids
        percentages  – proportion of pixels per cluster (%)
        kmeans       – fitted KMeans object (for metadata)
    """
    # Resize for speed; preserves colour distribution well enough
    img = image.convert("RGB").resize((200, 200), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(
        n_clusters=n_colors,
        init="k-means++",   # smarter centroid init
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    kmeans.fit(pixels)

    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=n_colors)
    percentages = counts / len(labels) * 100

    # Sort dominant → least dominant
    order = np.argsort(percentages)[::-1]
    colors = kmeans.cluster_centers_[order].astype(int)
    percentages = percentages[order]

    return colors, percentages, kmeans


def make_palette_figure(colors, percentages):
    """
    Horizontal stacked bar — widths proportional to dominance.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 1.6))
    fig.patch.set_facecolor("#0f0f11")
    ax.set_facecolor("#0f0f11")

    left = 0.0
    for color, pct in zip(colors, percentages):
        ax.barh(
            0,
            pct,
            left=left,
            height=1,
            color=np.array(color) / 255,
        )
        # Label inside bar if wide enough
        if pct > 8:
            ax.text(
                left + pct / 2,
                0,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=get_text_color(color),
                fontweight="bold",
            )
        left += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>ChromaLens</h1>
  <p>Upload an image · K-Means extracts your dominant colors</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONTROLS
# ─────────────────────────────────────────
col_upload, col_k = st.columns([3, 1], gap="medium")

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop or browse an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with col_k:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    n_colors = st.slider("Jumlah warna (K)", min_value=2, max_value=10, value=5)

# ─────────────────────────────────────────
#  MAIN LOGIC
# ─────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ── Image preview
    st.markdown('<div class="section-label">Gambar Input</div>', unsafe_allow_html=True)
    st.image(image, use_column_width=True)

    # ── Run K-Means
    with st.spinner("⚙️  Menjalankan K-Means Clustering…"):
        colors, percentages, kmeans = extract_colors(image, n_colors)

    # ── Palette bar
    st.markdown('<div class="section-label">Palet Warna Dominan</div>', unsafe_allow_html=True)
    fig_bar = make_palette_figure(colors, percentages)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # ── Swatch cards
    swatch_cols = st.columns(n_colors)
    for i, (col, pct) in enumerate(zip(colors, percentages)):
        hex_val = rgb_to_hex(col)
        txt_col = get_text_color(col)
        with swatch_cols[i]:
            st.markdown(f"""
            <div class="swatch-card">
              <div class="swatch-block" style="background:{hex_val}"></div>
              <div class="swatch-info">
                <div class="swatch-hex" style="color:{hex_val}">{hex_val.upper()}</div>
                <div class="swatch-rgb">rgb({col[0]}, {col[1]}, {col[2]})</div>
                <div class="swatch-pct">{pct:.1f}% piksel</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── K-Means Detail
    st.markdown('<div class="section-label">Detail Proses K-Means</div>', unsafe_allow_html=True)

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Jumlah Cluster (K)", n_colors)
    meta_col2.metric("Iterasi", kmeans.n_iter_)
    meta_col3.metric("Inertia (WCSS)", f"{kmeans.inertia_:,.0f}")

    st.markdown("---")
    st.markdown("**Centroid tiap cluster (nilai RGB rata-rata piksel):**")

    for i, (col, pct) in enumerate(zip(colors, percentages)):
        hex_val = rgb_to_hex(col)
        border_col = hex_val

        # progress bar width
        bar_html = f"""
        <div class="kmeans-box" style="border-left-color:{hex_val}">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
            <div style="width:28px;height:28px;border-radius:8px;background:{hex_val};flex-shrink:0;"></div>
            <div>
              <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;">
                Cluster {i+1} — {hex_val.upper()}
              </span><br>
              <span class="kmeans-iter">
                Centroid: rgb({col[0]}, {col[1]}, {col[2]}) &nbsp;|&nbsp; {pct:.2f}% piksel
              </span>
            </div>
          </div>
          <div style="background:#222;border-radius:99px;height:6px;overflow:hidden;">
            <div style="width:{pct}%;height:6px;background:{hex_val};border-radius:99px;"></div>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

    # ── Elbow hint
    st.markdown('<div class="section-label">Tips: Elbow Method</div>', unsafe_allow_html=True)
    st.info(
        f"Kamu memilih **K = {n_colors}**. "
        "Coba geser slider ke kiri/kanan — perhatikan bagaimana warna berubah. "
        "Nilai K optimal biasanya ada di 'siku' grafik inertia vs K (elbow point)."
    )

else:
    # ── Empty state
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#333;">
      <div style="font-size:4rem;">🖼️</div>
      <div style="font-family:'Syne',sans-serif; font-size:1.1rem; margin-top:1rem; color:#444;">
        Upload gambar untuk memulai
      </div>
      <div style="font-size:0.8rem; color:#2a2a2a; margin-top:0.4rem;">
        JPG · PNG · WEBP didukung
      </div>
    </div>
    """, unsafe_allow_html=True)
