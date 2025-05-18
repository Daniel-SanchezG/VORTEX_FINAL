# Cargar librerías necesarias
library(rcarbon)

# Leer los datos
datos <- read.csv("20250318_C14forSPD_Pub.csv", 
                  stringsAsFactors = FALSE,
                  encoding = "latin1")

# Verificar las columnas disponibles
print(names(datos))

# Preparar los datos
datos_c14 <- data.frame(
  Site = datos$Site,
  CRA = datos$C14.BP,
  Error = datos$C14.SD
)

# Filtrar filas con NA
datos_c14 <- datos_c14[!is.na(datos_c14$CRA), ]

# Identificar sitios únicos
sites <- unique(datos_c14$Site)
print(sites)

# Configurar dispositivo de salida
png("SPD_sites_final.png", width = 2400, height = 3600, res = 300)

# Configurar layout
layout(matrix(1:length(sites), ncol=1))

# Configurar márgenes
# mar = c(inferior, izquierdo, superior, derecho)
par(mar = c(0.5, 0.5, 1.5, 0.5))
par(oma = c(5, 5, 1, 1))  # Márgenes externos

# Preparar para almacenar límites de x
all_xlim <- c(8000, 1)

# Crear gráficos
for(i in 1:length(sites)) {
  site <- sites[i]
  datos_sitio <- datos_c14[datos_c14$Site == site, ]
  
  # Calibrar fechas
  cal_sitio <- calibrate(datos_sitio$CRA, datos_sitio$Error, calCurves = 'intcal20')
  
  # Generar SPD
  spd_sitio <- spd(cal_sitio, timeRange = all_xlim)
  
  # Convertir datos para graficar manualmente
  x_values <- 1950 - spd_sitio$grid$calBP  # Convertir BP a BC/AD
  y_values <- spd_sitio$grid$PrDens
  y_values <- y_values / max(y_values)  # Normalizar
  
  # Crear gráfico vacío
  plot(x_values, y_values, type = "n",
       xlim = range(x_values),
       ylim = c(0, 1),
       xlab = "", ylab = "",
       xaxt = "n", yaxt = "n",
       main = site)
  
  # Añadir área gris
  polygon(c(x_values[1], x_values, x_values[length(x_values)]),
          c(0, y_values, 0),
          col = "gray80", border = NA)
  
  # Añadir línea
  lines(x_values, y_values, col = "gray50")
  
  # Añadir marco
  box()
  
  # Añadir eje X solo en el último gráfico
  if(i == length(sites)) {
    axis(1, cex.axis = 0.8)
  }
}

# Añadir etiquetas de ejes
mtext("Years BC/AD", side = 1, outer = TRUE, line = 3, cex = 1.2)
mtext("Summed Probability", side = 2, outer = TRUE, line = 3, cex = 1.2, las = 0)

# Cerrar dispositivo
dev.off()

cat("¡Gráfico guardado como SPD_sites_final.png!\n")