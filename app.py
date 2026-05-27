import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

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

  [data-testid="stFileUploader"] {
    background: #1a1a1f;
    border: 1.5px dashed #333;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #ff6b6b; }

  .swatch-card {
    background: #1a1a1f;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
    transition: transform 0.2s;
  }
  .swatch-card:hover { transform: translateY(-3px); }
  .swatch-block { height: 90px; width: 100%; }
  .swatch-info { padding: 0.75rem; text-align: center; }
  .swatch-hex { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; letter-spacing: 1px; }
  .swatch-rgb { font-size: 0.72rem; color: #666; margin-top: 2px; }
  .swatch-pct { font-size: 0.75rem; color: #aaa; margin-top: 4px; }

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

  .kmeans-box {
    background: #1a1a1f;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid;
  }
  .kmeans-iter { font-size: 0.8rem; color: #666; }

  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }

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
    r, g, b = [x / 255.0 for x in bg_rgb]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0f0f11" if luminance > 0.45 else "#f0ede8"


def extract_colors(image: Image.Image, n_colors: int = 5):
    img = image.convert("RGB").resize((200, 200), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(
        n_clusters=n_colors,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    kmeans.fit(pixels)

    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=n_colors)
    percentages = counts / len(labels) * 100

    order = np.argsort(percentages)[::-1]
    colors = kmeans.cluster_centers_[order].astype(int)
    percentages = percentages[order]
    # remap labels to new order
    label_map = {old: new for new, old in enumerate(order)}
    mapped_labels = np.array([label_map[l] for l in labels])

    return colors, percentages, kmeans, pixels, mapped_labels


def make_palette_figure(colors, percentages):
    fig, ax = plt.subplots(figsize=(10, 1.6))
    fig.patch.set_facecolor("#0f0f11")
    ax.set_facecolor("#0f0f11")

    left = 0.0
    for color, pct in zip(colors, percentages):
        ax.barh(0, pct, left=left, height=1, color=np.array(color) / 255)
        if pct > 8:
            ax.text(
                left + pct / 2, 0, f"{pct:.1f}%",
                ha="center", va="center", fontsize=9,
                color=get_text_color(color), fontweight="bold",
            )
        left += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def make_elbow_figure(pixels, max_k=10):
    """Compute inertia for K=1..max_k and plot the elbow curve."""
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=5,
                    max_iter=100, random_state=42)
        km.fit(pixels)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#1a1a1f")
    ax.set_facecolor("#1a1a1f")

    # Shaded area under curve
    ax.fill_between(list(k_range), inertias, alpha=0.12, color="#4d96ff")

    # Main line
    ax.plot(list(k_range), inertias, color="#4d96ff", linewidth=2.5,
            marker="o", markersize=7, markerfacecolor="#ffd93d",
            markeredgecolor="#0f0f11", markeredgewidth=1.5)

    # Annotate each point
    for k, inertia in zip(k_range, inertias):
        ax.annotate(
            f"K={k}",
            xy=(k, inertia),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#888",
        )

    ax.set_xlabel("Jumlah Cluster (K)", color="#aaa", fontsize=10)
    ax.set_ylabel("Inertia (WCSS)", color="#aaa", fontsize=10)
    ax.set_title("Elbow Method — Optimal K", color="#f0ede8",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#666")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333")
    ax.set_xticks(list(k_range))
    ax.grid(axis="y", color="#222", linewidth=0.8)

    plt.tight_layout()
    return fig


def make_scatter_figure(pixels, colors, labels, n_colors):
    """
    2-D PCA scatter of pixel clusters — mirrors the K-Means scatter
    plot from the lecture slides (halaman 6–7).
    Samples a subset of pixels for speed.
    """
    # Sample max 3000 pixels so rendering is fast
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pixels), size=min(3000, len(pixels)), replace=False)
    sample_px = pixels[idx]
    sample_lb = labels[idx]

    # PCA → 2D
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(sample_px)
    centroids_2d = pca.transform(colors.astype(np.float32))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a1f")
    ax.set_facecolor("#1a1a1f")

    for i in range(n_colors):
        mask = sample_lb == i
        cluster_color = np.array(colors[i]) / 255
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=[cluster_color],
            s=8, alpha=0.55, linewidths=0,
        )

    # Draw centroids as big stars
    for i, (cx, cy) in enumerate(centroids_2d):
        cluster_color = np.array(colors[i]) / 255
        hex_c = rgb_to_hex(colors[i])
        ax.scatter(cx, cy, s=220, c=[cluster_color],
                   marker="*", edgecolors="#0f0f11", linewidths=1.2, zorder=5)
        ax.annotate(
            f" C{i+1}\n {hex_c.upper()}",
            xy=(cx, cy), fontsize=7, color="#ddd",
            fontweight="bold", zorder=6,
        )

    ax.set_title(
        "Scatter Plot Cluster (PCA 2D)",
        color="#f0ede8", fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Principal Component 1", color="#666", fontsize=9)
    ax.set_ylabel("Principal Component 2", color="#666", fontsize=9)
    ax.tick_params(colors="#444")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#333")
    ax.grid(color="#222", linewidth=0.6)

    plt.tight_layout()
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

    st.markdown('<div class="section-label">Gambar Input</div>', unsafe_allow_html=True)
    st.image(image, use_column_width=True)

    with st.spinner("⚙️  Menjalankan K-Means Clustering…"):
        colors, percentages, kmeans, pixels, labels = extract_colors(image, n_colors)

    # ── Palette bar
    st.markdown('<div class="section-label">Palet Warna Dominan</div>', unsafe_allow_html=True)
    fig_bar = make_palette_figure(colors, percentages)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # ── Swatch cards
    swatch_cols = st.columns(n_colors)
    for i, (col, pct) in enumerate(zip(colors, percentages)):
        hex_val = rgb_to_hex(col)
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

    # ════════════════════════════════════════
    #  VISUALISASI K-MEANS (NEW!)
    # ════════════════════════════════════════
    st.markdown('<div class="section-label">Visualisasi K-Means</div>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2, gap="medium")

    with viz_col1:
        st.markdown("**📈 Elbow Method**")
        st.caption("Grafik inertia vs K untuk menemukan jumlah cluster optimal (seperti di materi hal. 9)")
        with st.spinner("Menghitung elbow curve…"):
            fig_elbow = make_elbow_figure(pixels, max_k=10)
        st.pyplot(fig_elbow, use_container_width=True)
        plt.close(fig_elbow)

    with viz_col2:
        st.markdown("**🔵 Scatter Plot Cluster**")
        st.caption("Visualisasi 2D tiap piksel dalam cluster-nya (PCA projection)")
        with st.spinner("Membuat scatter plot…"):
            fig_scatter = make_scatter_figure(pixels, colors, labels, n_colors)
        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

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

else:
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
