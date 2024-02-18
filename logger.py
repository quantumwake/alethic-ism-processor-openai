# The Alethic Instruction-Based State Machine (ISM) is a versatile framework designed to 
# efficiently process a broad spectrum of instructions. Initially conceived to prioritize
# animal welfare, it employs language-based instructions in a graph of interconnected
# processing and state transitions, to rigorously evaluate and benchmark AI models
# apropos of their implications for animal well-being. 
# 
# This foundation in ethical evaluation sets the stage for the framework's broader applications,
# including legal, medical, multi-dialogue conversational systems.
# 
# Copyright (C) 2023 Kasra Rasaee, Sankalpa Ghose, Yip Fai Tse (Alethic Research) 
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# 
import logging as log
import dotenv
import os

dotenv.load_dotenv()

# log_level = os.environ.get("LOG_LEVEL", "DEBUG")
# formatter = log.Formatter('[%(asctime)s] [%(process)d] [%(thread)d] [%(levelname)s] %(name)s: %(message)s')
# stream_handler = log.StreamHandler(sys.stdout)
# stream_handler.setFormatter(formatter)
# logging = log.getLogger(__name__)
# logging.addHandler(stream_handler)
# logging.setLevel(log_level)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging = log.getLogger(__name__)
log.basicConfig(encoding='utf-8', level=LOG_LEVEL)