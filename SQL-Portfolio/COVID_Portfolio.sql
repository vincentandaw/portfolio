-- COVID DEATHS DATASET
SELECT *
FROM PortfolioProject.dbo.CovidDeaths
WHERE continent is not null
ORDER BY 3,4

--SELECT *
--FROM PortfolioProject.dbo.CovidVaccination
--ORDER BY 3,4

-- Select Data Used
SELECT Location, date, total_cases, new_cases, total_deaths, population
FROM PortfolioProject.dbo.CovidDeaths
ORDER BY 1,2

-- Looking at Total Cases vs. Total Deaths
-- Shows likelihood of dying if you contract Covid-19 in Indonesia
SELECT Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
FROM PortfolioProject.dbo.CovidDeaths
WHERE location like 'Indonesia'
ORDER BY 1,2

-- Looking at Total Cases vs Population
-- Shows the Covid-to-Population Ratio in Indonesia
SELECT Location, date, total_cases, population, (total_cases/population)*100 as CovidPopRatio
FROM PortfolioProject.dbo.CovidDeaths
WHERE location like 'Indonesia'
ORDER BY 1,2

--  Looking at Countries with Highest Infection Rate vs. Population
SELECT Location, population, MAX(total_cases) AS HighestInfectionCount, (MAX(total_cases)/population)*100 as PercentPopInfected
FROM PortfolioProject.dbo.CovidDeaths
--WHERE location like 'Indonesia'
GROUP BY location,population
ORDER BY PercentPopInfected DESC

-- Breaking Down by Continent
SELECT continent, MAX(cast(total_deaths as int)) AS TotalDeathCount
FROM PortfolioProject.dbo.CovidDeaths
--WHERE location like 'Indonesia'
WHERE continent is not null
GROUP BY continent
ORDER BY TotalDeathCount DESC

-- Showing Countries with Highest Death Count per Population (not null)
SELECT Location, MAX(cast(total_deaths as int)) AS TotalDeathCount
FROM PortfolioProject.dbo.CovidDeaths
--WHERE location like 'Indonesia'
WHERE continent is not null
GROUP BY location
ORDER BY TotalDeathCount DESC

-- Showing Countries with Highest Death Count per Population (with null)
SELECT Location, MAX(cast(total_deaths as int)) AS TotalDeathCount
FROM PortfolioProject.dbo.CovidDeaths
--WHERE location like 'Indonesia'
where continent is null
GROUP BY location
ORDER BY TotalDeathCount DESC

-- GLOBAL NUMBERS
SELECT date, SUM(new_cases) as total_cases, SUM(CAST(new_deaths as int)) as total_deaths, SUM(CAST(new_deaths as int))/SUM(new_cases)*100 as GlobalDeathsPercentage
FROM PortfolioProject.dbo.CovidDeaths
WHERE continent is not null 
GROUP BY date
ORDER BY 1,2

-- Total Death Percentage WorldWide
SELECT SUM(new_cases) as total_cases, SUM(CAST(new_deaths as int)) as total_deaths, SUM(CAST(new_deaths as int))/SUM(new_cases)*100 as GlobalDeathsPercentage
FROM PortfolioProject.dbo.CovidDeaths
WHERE continent is not null 
ORDER BY 1,2


-- COVID VACCINATION DATASET
SELECT *
FROM PortfolioProject.dbo.CovidVaccination
ORDER BY 3,4

-- JOINING COVID DEATH AND COVID VACCINATION DATASET
-- Total Population vs. Vaccinations
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, 
SUM(CAST(vac.new_vaccinations as bigint)) OVER (PARTITION BY dea.location ORDER BY dea.location, dea.date)as RollingPeopleVaccinated--, (RollingPeopleVaccinated/population)*100
FROM PortfolioProject.dbo.CovidDeaths dea
JOIN PortfolioProject.dbo.CovidVaccination vac
	ON dea.location = vac.location
	and dea.date = vac.date
WHERE dea.continent is not null
ORDER BY 2,3

-- USE CTE
;With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
FROM PortfolioProject.dbo.CovidDeaths dea
JOIN PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
)
Select *, (RollingPeopleVaccinated/Population)*100 as RollingRatio
From PopvsVac

-- TEMP TABLE
DROP TABLE if exists #PercentPopulationVaccinated
CREATE TABLE #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
new_vaccinations numeric,
RollingPeopleVaccinated numeric
)
INSERT INTO #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
FROM PortfolioProject.dbo.CovidDeaths dea
JOIN PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
Select *, (RollingPeopleVaccinated/Population)*100 as RollingRatio
From #PercentPopulationVaccinated


-- Creating View to Store Data for Later Visualiztions
CREATE VIEW PercentPopulationVaccinated AS
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
FROM PortfolioProject.dbo.CovidDeaths dea
JOIN PortfolioProject.dbo.CovidVaccination vac
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null

-- TESTING CREATEVIEW
SELECT *
FROM PercentPopulationVaccinated